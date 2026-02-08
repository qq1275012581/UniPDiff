from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import trunc_normal_

from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusionmodule import DetNet, AttUnet
from utils_tool.data_utils import   make_beta_schedule
from utils_tool.transform import default, extract_into_tensor


class ResCast(nn.Module):
    """
    Args:
        root_dir (str): file save path
        img_size (list): image size of the input data
        embed_dim (int): embedding dimension
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        drop_rate (float): dropout rate
    """
    def __init__(
        self, 
        var_names=None,
        embed_dim=384, 
        beta_schedule= "linear",
        timesteps=1000,
        given_betas=None,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization='eps',
        v_posterior=0.,
        scale_by_std=True,
        scale_factor=1.,
        T_in=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.v_posterior = v_posterior
        
        self.weight = None

        
        # DDPM
        # --------------------------------------------------------------------------
        self.latent_model = AttUnet(
            dim=embed_dim,
            T_in=T_in,
            dim_mults=(1,2,4,8),
            channels=len(var_names)
        )

        self.contxt_net = DetNet(
            dim=embed_dim,
            dim_mults=(1,2,4,8),
            channels=len(var_names)
        )

        self.parameterization = parameterization

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        
        self.scale_by_std = scale_by_std
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
    
        self.initialize_weights()
    
    def extract_into_tensor(self, a, t, x_shape):
        return extract_into_tensor(a=a, t=t, x_shape=x_shape,
                                   batch_axis=0)

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def initialize_weights(self):

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def apply_model(self, x_noisy, t, cond, ctx=None, idx=None):
        # print("apply model", x_noisy.shape, t.shape, cond.shape)
        x_recon = self.latent_model(x_noisy, t, cond, ctx=ctx, idx=idx)
        # print("x_recon", x_recon.shape)
        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def forward(self, x, y):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            hours: `[B]` shape. Forecasting absolution times of each element of the batch.
            region_info: Containing the region's information

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        b, T, v, h, w = x.size()

        t = torch.randint(0, self.num_timesteps, (b,), device=y.device).long()

        y = rearrange(y, "(b t) c h w -> b t c h w", t=1)

        global_ctx, local_ctx = self.contxt_net.scan_ctx(x)

        loss = self.p_losses(y, x, t, noise=None, 
                                 ctx=local_ctx, 
                                 idx=torch.full((b,), 0, device = y.device, dtype = torch.long))

        return loss, local_ctx

    def evaluate(self, x):
        b, v, h, w = x.size()

        zc = rearrange(x, "(b t) c h w -> b t c h w", t=1)
        global_ctx, local_ctx = self.contxt_net.scan_ctx(zc)

        preds = self.sample(
            cond=x, ctx=local_ctx, 
            idx=torch.full((b,), 0, device = x.device, dtype = torch.long),
            return_intermediates=False,
            verbose=False)
        
        return preds

    # 修改 evaluate 函数以使用新采样器
    def evaluate_ddim(self, x, steps=50, ensemble=False, num_ensembles=50):
        b, t, v, h, w = x.size()
        # zc = rearrange(x, "(b t) c h w -> b t c h w", t=1)
        global_ctx, local_ctx = self.contxt_net.scan_ctx(x)
        
        preds = self.ddim_sample(
            cond=x, ctx=local_ctx, 
            idx=torch.full((b,), 0, device=x.device, dtype=torch.long),
            ddim_steps=steps, ensemble=ensemble, num_ensembles=num_ensembles
        )
        return preds
    

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    
    def get_loss(self, pred, target, mean=True):
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')

        return loss
    
    def p_losses(self, x_start, cond, t, noise=None, ctx=None, idx=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print(x_noisy.shape, x_start.shape, noise.shape)
        model_output = self.apply_model(x_noisy, t, cond, ctx=ctx, idx=idx)
        # print(model_output.shape)
        # TODO: add v-prediction
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target)

        return loss_simple
    

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self.extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, zt, zc, t, ctx=None, idx=None,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        
        
        model_out = self.apply_model(zt, t_in, zc, ctx=ctx, idx=idx)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, zt, t, zc, **corrector_kwargs)

        if self.parameterization == "eps":
            z_recon = self.predict_start_from_noise(zt, t=t, noise=model_out)
        elif self.parameterization == "x0":
            z_recon = model_out
        else:
            raise NotImplementedError()
        
        # z_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=z_recon, x_t=zt, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, z_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self, zt, zc, t, ctx=None, idx=None,
                return_x0=False, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        batch_size = zt.shape[0]
        device = zt.device
        outputs = self.p_mean_variance(zt=zt, zc=zc, t=t, ctx=ctx, idx=idx,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = torch.randn(zt.shape, device=device) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask_shape = [1, ] * len(zt.shape)
        nonzero_mask_shape[0] = batch_size
        nonzero_mask = (1 - (t == 0).float()).reshape(*nonzero_mask_shape)

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, ctx=None, idx=None,
                      return_intermediates=False, timesteps=None):

        device = self.betas.device
        batch_size = shape[0]

        img = torch.randn(shape, device=device)

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps


        for i in tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps, disable=False):
            ts = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            img = self.p_sample(zt=img, zc=cond, t=ts, ctx=ctx, idx=idx)
            # break


        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, ctx=None, idx=None, use_alignment=False, alignment_kwargs=None,
               return_intermediates=False, timesteps=None, **kwargs):
        
        zc = rearrange(cond, "(b t) c h w -> b t c h w", t=1)
        shape = zc[:, :, 0:1].shape
        output = self.p_sample_loop(
            cond=zc, shape=shape, ctx=ctx, idx=idx,
            return_intermediates=return_intermediates, timesteps=timesteps)
        
        # output.clamp_(-1., 1.)
        output = rearrange(output, "b t c h w -> (b t) c h w")
        # output.clamp_(-1., 1.)
    
        return output

    # --- 新增：DDIM 采样逻辑 ---
    @torch.no_grad()
    def ddim_sample(self, cond, ctx=None, idx=None, ddim_steps=50, ddim_eta=0.0, ensemble=False, num_ensembles=50):
        device = self.alphas_cumprod.device
        b = cond.shape[0]
        
        # 1. 处理条件输入 zc
        # 将 cond 从 (b*t, c, h, w) 重排为 (b, t, c, h, w)
        # zc = rearrange(cond, "(b t) c h w -> b t c h w", t=1)
        zc = cond
        
        # 确定初始噪声的 shape
        # 原始 shape 可能是 (b, 1, 1, h, w)
        shape = zc[:, 0:1, 0:1].shape 
        
        if ensemble:
            # 扩展采样目标的形状: (b * e, t, c, h, w)
            shape = (b * num_ensembles,) + shape[1:]
            
            # 【关键：同步扩展条件场】
            # zc 从 (b, t, c, h, w) 变为 (b*e, t, c, h, w)
            # 注意：使用 repeat_interleave 保证顺序是 [b1, b1, b1, ..., b2, b2, ...]
            zc = zc.repeat_interleave(num_ensembles, dim=0)
            
            if ctx is not None:
                # 假设 ctx 是 (b, ...) 维度的张量
                ctx = ctx.repeat_interleave(num_ensembles, dim=0)
            if idx is not None:
                # 假设 idx 是 (b, ...) 维度的张量
                idx = idx.repeat_interleave(num_ensembles, dim=0)
                
            current_b = b * num_ensembles
        else:
            current_b = b

        # 2. 构建时间步序列
        times = torch.linspace(-1, self.num_timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        # print(time_pairs)
        # 初始化噪声图像
        img = torch.randn(shape, device=device)

        denoise_img_list = []
        # 3. 迭代采样
        for i, j in tqdm(time_pairs, desc='DDIM Ensemble Sampling', disable=False):
            # 时间步张量需要匹配当前的 Batch Size (current_b)
            t = torch.full((current_b,), i, device=device, dtype=torch.long)
            t_next = torch.full((current_b,), j, device=device, dtype=torch.long)

            # 获取 alpha_bar (extract_into_tensor 会处理维度对齐)
            alpha_bar = self.extract_into_tensor(self.alphas_cumprod, t, img.shape)
            if j < 0:
                alpha_bar_next = torch.ones_like(alpha_bar)
            else:
                alpha_bar_next = self.extract_into_tensor(self.alphas_cumprod, t_next, img.shape)

            # 模型预测噪声
            # 此时 img, t, zc, ctx 的第 0 维都是 current_b
            epsilon_theta = self.apply_model(img, t, zc, ctx=ctx, idx=idx)

            # 预测 x0
            pred_x0 = (img - torch.sqrt(1 - alpha_bar) * epsilon_theta) / torch.sqrt(alpha_bar)
            
            # 计算 sigma (控制随机性)
            # 如果 ddim_eta = 0，则 sigma = 0，采样变为确定性
            sigma = ddim_eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_next)
            
            # 计算指向 xt 的方向部分
            pred_dir_xt = torch.sqrt(1 - alpha_bar_next - sigma**2) * epsilon_theta
            
            # 计算 x_{t-1}
            x_prev = torch.sqrt(alpha_bar_next) * pred_x0 + pred_dir_xt
            
            if ddim_eta > 0:
                noise = torch.randn_like(img)
                x_prev = x_prev + sigma * noise
            # print(i, j)
            img = x_prev
            if (i+1) % 100 == 0:
                denoise_img_list.append(rearrange(img, "b t c h w -> (b t) c h w"))

        # 4. 后处理结果
        if ensemble:
            # 将结果从 (b*e, t, c, h, w) 还原为 (b, e, t, c, h, w)
            img = rearrange(img, "(b e) t c h w -> b e t c h w", e=num_ensembles)
            
            # 你可以选择：
            # A. 返回集合平均值 (b, t, c, h, w)
            img_out = img.mean(dim=1) 
            
            # B. 如果需要计算概率或保留成员，也可以直接返回 6D 张量，此处按你原逻辑取 mean
        else:
            img_out = img # (b, t, c, h, w)
        denoise_img_list.append(rearrange(img_out, "b t c h w -> (b t) c h w"))

        # 还原为原代码期望的 (b*t, c, h, w) 格式
        return rearrange(img_out, "b t c h w -> (b t) c h w"), denoise_img_list
    
    
def get_var_names(surf_vars, atmo_vars, levels):
    var_names = surf_vars.copy()
    for var in atmo_vars:
        for level in levels:
            var_names.append(f"{var}_{level}")
    return var_names


if __name__ == '__main__':
    tp_var = 'tp_gpm'
    era5_surface_vars = ['t2m', 'msl', 'u10', 'v10']
    era5_upper_vars = ['t', 'z', 'u', 'v', 'q']
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    var_names = get_var_names(era5_surface_vars, era5_upper_vars, levels)

    rescast = ResCast(
        var_names=var_names,
        embed_dim=384,
        T_in=2
    ).to(device)

    x = torch.randn(1, 2, len(var_names), 120, 240).to(device)
    y = torch.randn(1, 1, 120, 240).to(device)

    loss, ctx = rescast(x, y)
    print(loss)
    preds = rescast.evaluate_ddim(x, steps=50)
    print(preds.shape)
