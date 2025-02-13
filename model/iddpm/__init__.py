from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        device,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            device=device    # note
        )


    def sample(
        self,
        model,
        z,
        device,
        additional_args=None,
        mask=None,
        progress=True,
        ob=None,
        measurement_cond_fn=None,
    ):
        model_args = {}
        model_args["y"] = None
        if additional_args is not None:
            model_args.update(additional_args)
        samples = self.p_sample_loop(
            model,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=progress,
            device=device,
            mask=mask,
            observation=ob,
            measurement_cond_fn=measurement_cond_fn,
        )
        return samples

