import numpy as np
import torch
import torch.nn as nn
from diffmodels_or import diff_FSDI
import math
import torch.nn.functional as F
import logging

def rfft_half(x): return torch.fft.rfft(x, dim=-1, norm="ortho")
def irfft_half(X, L): return torch.fft.irfft(X, n=L, dim=-1, norm="ortho")
def white_complex_half(shape, L, device):
    z = torch.randn(*shape, 2, device=device) / math.sqrt(2.0)
    z = torch.view_as_complex(z)
    z[..., 0] = torch.randn(*shape[:-1], device=device)  
    if L % 2 == 0: z[..., -1] = torch.randn(*shape[:-1], device=device)  
    return z

def linear_ramp(step, warmup, maxv):
    if warmup<=0: return float(maxv)
    r = min(1.0, step/float(max(1,warmup)))
    return float(maxv)*r


@torch.no_grad()
def inverse_s2(w, s2_max=None):
    Lf = w.numel()
    invw = 1.0/(w+1e-12)
    s2 = invw / invw.sum() * Lf
    return s2

class FSDI_base(nn.Module):

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        # ---- backbone ----
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional: self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(self.target_dim, self.emb_feature_dim)

        cfg = dict(config["diffusion"])
        cfg["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_FSDI(cfg, input_dim)

        # ---- scalar schedule (baseline) ----
        self.num_steps = cfg["num_steps"]
        if cfg["schedule"] == "quad":
            beta = np.linspace(cfg["beta_start"]**0.5, cfg["beta_end"]**0.5, self.num_steps)**2
        elif cfg["schedule"] == "linear":
            beta = np.linspace(cfg["beta_start"], cfg["beta_end"], self.num_steps)
        else:
            raise ValueError("unknown schedule")
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)     # (T,)
        self.alpha_hat = 1.0 - self.beta                                       # (T,)
        self.alpha = torch.cumprod(self.alpha_hat, dim=0)                      # (T,)
        self.alpha_torch = self.alpha.view(-1,1,1)                             # for baseline path
        self.alpha_bar_base = self.alpha.clone()

        # ---- curriculum / opts ----
        self.global_step = 0
        self.use_ddim = bool(cfg.get("use_ddim", False))  
        self.eps_floor = float(cfg.get("eps_floor", 1e-6))

        # ---- freq shaping ----
        self.use_freq_noise = bool(cfg.get("freq_noise", False))
        if self.use_freq_noise:
            self.L = int(cfg["seq_len"])
            self.Lf = self.L//2 + 1

            w0 = torch.ones(self.Lf, device=device)
            if self.Lf > 1:
                w0[1:self.Lf-1] = 2.0
            if self.L % 2 == 0:
                w0[-1] = 1.0
            self.register_buffer("w0_parseval", w0)

            def _normalize_s2_time_energy(s2: torch.Tensor) -> torch.Tensor:

                scale = self.L / torch.clamp((s2 * self.w0_parseval).sum(), min=1e-12)
                return s2 * scale

            s_vec = cfg.get("s_vec", None)
            s = torch.tensor(np.asarray(s_vec, np.float32), device=device)
            assert s.numel()==self.Lf

            s = s / (s.mean()+1e-8)
            w = s.clone()                            
            self.register_buffer("w_importance", w)

  
            s2_uniform = torch.ones(self.Lf, device=device)
            s2_uniform = s2_uniform / s2_uniform.sum() * self.Lf
            s2_uniform = _normalize_s2_time_energy(s2_uniform)
          

            s2_wf = inverse_s2(w, s2_max=None)
            s2_wf = _normalize_s2_time_energy(s2_wf)


            self.register_buffer("s2_uniform", s2_uniform)             # (Lf,)
            self.register_buffer("s2_wf", s2_wf)                       # (Lf,)


            V_uni, V_wf = [], []
            v = torch.zeros(self.Lf, device=device)
            for t in range(self.num_steps):
                v = self.alpha_hat[t]*v + self.beta[t]*s2_uniform
                V_uni.append(v.clone())
            v = torch.zeros(self.Lf, device=device)
            for t in range(self.num_steps):
                v = self.alpha_hat[t]*v + self.beta[t]*s2_wf
                V_wf.append(v.clone())
            self.register_buffer("V_uni", torch.stack(V_uni,0))        # (T,Lf)
            self.register_buffer("V_wf", torch.stack(V_wf,0))          # (T,Lf)





    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1/torch.pow(10000.0, torch.arange(0,d_model,2).to(self.device)/d_model)
        pe[:,:,0::2] = torch.sin(position*div_term); pe[:,:,1::2] = torch.cos(position*div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand = torch.rand_like(observed_mask) * observed_mask
        rand = rand.reshape(len(rand), -1)
        for i in range(len(observed_mask)):
            ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * ratio)
            rand[i][rand[i].topk(num_masked).indices] = -1
        return (rand>0).reshape(observed_mask.shape).float()

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None: for_pattern_mask = observed_mask
        if self.target_strategy=="mix":
            rand_mask = self.get_randmask(observed_mask)
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            if self.target_strategy=="mix" and np.random.rand()>0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i]*for_pattern_mask[i-1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        B,K,L = cond_mask.shape
        te = self.time_embedding(observed_tp, self.emb_time_dim).unsqueeze(2).expand(-1,-1,K,-1)
        fe = self.embed_layer(torch.arange(self.target_dim).to(self.device)).unsqueeze(0).unsqueeze(0).expand(B,L,-1,-1)
        side = torch.cat([te,fe], dim=-1).permute(0,3,2,1)
        if not self.is_unconditional: side = torch.cat([side, cond_mask.unsqueeze(1)], dim=1)
        return side


    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum=0
        for t in range(self.num_steps):
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum/self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B,K,L = observed_data.shape
        if is_train!=1: t = (torch.ones(B, dtype=torch.long, device=self.device)*set_t)
        else: t = torch.randint(0, self.num_steps, [B], device=self.device)

        if not self.use_freq_noise:
            current_alpha = self.alpha_torch[t]
            noise = torch.randn_like(observed_data)
            x_t = current_alpha.sqrt()*observed_data + torch.sqrt(1.0-current_alpha)*noise
            total_input = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)
            predicted = self.diffmodel(total_input, side_info, t)
            target_mask = observed_mask - cond_mask
            residual = (noise - predicted) * target_mask
            denom = target_mask.sum()
            if is_train==1: self.global_step += 1
            return (residual**2).sum() / (denom if denom>0 else 1.0)

        assert L==self.L
        Lf = self.Lf
        gamma = 1
        V_t = (1-gamma)*self.V_uni[t,:Lf] + gamma*self.V_wf[t,:Lf]      # (Lf,)

        V_t = V_t.unsqueeze(1).expand(B,K,Lf)
        V_safe = torch.clamp(V_t, min=self.eps_floor)


        X0 = rfft_half(observed_data)
        Zstar = white_complex_half(X0.shape, L, self.device)
        Xt = torch.sqrt(self.alpha_bar_base[t]).view(B,1,1)*X0 + torch.sqrt(V_safe)*Zstar
        x_t = irfft_half(Xt, L)


        total_input = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)
        eps_hat_time = self.diffmodel(total_input, side_info, t)              # (B,K,L)


        scale_std     = torch.sqrt(1.0 - self.alpha_bar_base[t]).view(B,1,1)  
        eps_std_true  = irfft_half(Zstar*torch.sqrt(V_safe), L) / scale_std                    

        U_hat = rfft_half(eps_hat_time * scale_std)/torch.sqrt(V_safe)                        
        loss_U = ((U_hat.real - Zstar.real)**2 + (U_hat.imag - Zstar.imag)**2).mean()

        target_mask = (observed_mask - cond_mask)
        denom       = target_mask.sum()
        loss_time   = (((eps_hat_time - eps_std_true) * target_mask)**2).sum() / (denom if denom>0 else 1.0)

        loss = 0.05*loss_U + 1*loss_time
        if is_train==1: self.global_step += 1
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional: return noisy_data.unsqueeze(1)
        cond_obs = (cond_mask*observed_data).unsqueeze(1)
        noisy_target = ((1-cond_mask)*noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)


    @torch.no_grad()
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B,K,L = observed_data.shape
        imputed = torch.zeros(B, n_samples, K, L, device=self.device)

        if not self.use_freq_noise:

            for i in range(n_samples):
                current = torch.randn_like(observed_data)
                for t in range(self.num_steps-1, -1, -1):
                    cond_obs = (cond_mask*observed_data).unsqueeze(1)
                    noisy_target = ((1-cond_mask)*current).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                    pred = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device))
                    coeff1 = 1/self.alpha_hat[t].sqrt()
                    coeff2 = (1-self.alpha_hat[t])/(1-self.alpha[t]).sqrt()
                    current = coeff1*(current - coeff2*pred)
                    if t>0 and not self.use_ddim:
                        sigma = (((1.0-self.alpha[t-1])/(1.0-self.alpha[t]))*self.beta[t]).sqrt()
                        current += sigma*torch.randn_like(current)
                imputed[:,i]=current
            return imputed

        assert L==self.L
        Lf = self.Lf
        gamma = 1
        V_all = (1-gamma)*self.V_uni + gamma*self.V_wf                   
        s2_all = (1-gamma)*self.s2_uniform + gamma*self.s2_wf             

        for i in range(n_samples):
            current = irfft_half(torch.sqrt(s2_all.view(1,1,-1).clamp_min(self.eps_floor))*white_complex_half((B,K,Lf), L, self.device), L)
            for t in range(self.num_steps-1, -1, -1):
                cond_obs = (cond_mask*observed_data).unsqueeze(1)
                noisy_target = ((1-cond_mask)*current).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)

                eps_hat_time = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device))

                eps_hat_f = rfft_half(eps_hat_time*torch.sqrt(1-self.alpha_bar_base[t]))
                Xt_f = rfft_half(current)

                ab_t = float(self.alpha_bar_base[t].item())
                ab_tm1 = float(self.alpha_bar_base[t-1].item()) if t>0 else 1.0
                at = ab_t/ab_tm1 
                V_t = V_all[t,:Lf].unsqueeze(0).unsqueeze(0).expand(B,K,Lf)
                V_tm1 = (V_all[t-1,:Lf] if t>0 else torch.zeros_like(V_all[0,:Lf])).unsqueeze(0).unsqueeze(0).expand(B,K,Lf)
                V_t = torch.clamp(V_t, min=self.eps_floor)
                beta_t = self.beta[t]
                sigma2_t = beta_t * s2_all[:Lf]  # (Lf,)
                sigma2_t = sigma2_t.unsqueeze(0).unsqueeze(0).expand(B,K,Lf)


                U_hat = eps_hat_f / torch.sqrt(V_t)
                X0_hat = (Xt_f - torch.sqrt(V_t)*U_hat) / math.sqrt(ab_t)


                mu = math.sqrt(ab_tm1)*X0_hat + math.sqrt(at)*(V_tm1/V_t)*(Xt_f - math.sqrt(ab_t)*X0_hat)

  
                if t>0 and not self.use_ddim:
                    Sigma_post = V_tm1 * (sigma2_t / V_t)     
                    noise_f = white_complex_half(Xt_f.shape, L, self.device)
                    Xprev_f = mu + torch.sqrt(torch.clamp(Sigma_post, min=0.0)) * noise_f
                else:
                    Xprev_f = mu

                current = irfft_half(Xprev_f, L)

            imputed[:,i]=current
        return imputed

    def forward(self, batch, is_train=1):
        (observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, _) = self.process_data(batch)
        if is_train==0:
            cond_mask = gt_mask
        elif self.target_strategy!="random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train==1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (observed_data, observed_mask, observed_tp, gt_mask, _, cut_length) = self.process_data(batch)
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

class FSDI_PM25(FSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(FSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class FSDI_Physio(FSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(FSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )



class FSDI_Forecasting(FSDI_base):
    def __init__(self, config, device, target_dim):
        super(FSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp