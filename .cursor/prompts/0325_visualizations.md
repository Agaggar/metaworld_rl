The basic functionality of the repo works pretty well. I did some quick tests to see reward over episodes for 5 environments by calling `python3.10 main_subprocess.py`

After training a policy for different environments, I want to do the following:
1. create a video animation of executing the policy for a fixed number of timesteps. Some similar code that I have used to plot goes like this:
def render_video(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None):
        """
        Visualize trajectories using the trainer's planner/rollout logic.

        Args:
            trainer: Trainer instance (must expose env, model, device, past_length, pred_length/config).
            run_path: Path to save the resulting video.
            max_steps: Number of environment steps to visualize.
            closed_loop: If True, feed ground-truth observations back into the model window.
                         If False, feed the model's predicted next frame (open-loop rollout).
        """
        model = trainer.model
        device = trainer.device
        trainer.env.unwrapped.max_path_length = 5000
        env = trainer.env
        past_len = model.past_length if hasattr(model, 'past_length') else 3
        pred_len = model.pred_length if hasattr(model, 'pred_length') else 3
        try:
            closed_loop_policy = trainer.config['closed_loop']['policy']
        except:
            print('No closed loop policy found in trainer config, defaulting to random')
            closed_loop_policy = 'random'

        saved_state = torch.zeros((max_steps, 13), device='cpu')
        if env_reset_seed is None:
            env_reset_seed = np.random.randint(0, 1e2)

        model.eval()
        obs, _ = env.reset(seed=env_reset_seed)
        frame_buffer = []   # frames used as model input window
        true_frames = []    # ground-truth frames for visualization
        recon_frames = []   # model reconstructions
        pred_sequences = [] # list of predicted sequences per step
        plan_obj_vals = []  # objective function values per step
        env_rew = []        # env rewards [blind during training]
        contacts = []      # env contact info [blind during training], also doesn't work yet
        # Counting contacts
        mj_model = env.unwrapped.model
        mj_data = env.unwrapped.data
        robot_geom, obj_geom = get_mujoco_geom_keys_index(self.dataset_name)

        # Prime buffer with the first observation
        first_render_raw = env.render()  # numpy array (H, W, C) [0-255]
        first_img = process_image(first_render_raw, self.dataset_name, downscale=False).permute(2, 0, 1)
        for _ in range(past_len):
            frame_buffer.append(process_image(first_render_raw, self.dataset_name, downscale=True).permute(2, 0, 1))

        step_idx = 0
        if closed_loop_policy in ['informative', 'maxdyn']:
            trainer._init_cem_mu_sig()
            mu = trainer.init_control.clone()
            sigma = trainer.sigma.clone()
        for step_idx in tqdm(range(max_steps), desc="Visualizing Planner timesteps"): # range(max_steps): #
            # Current frame (ground truth)
            curr_render_raw = env.render()  # Store raw render
            curr_img = process_image(curr_render_raw, self.dataset_name, downscale=False).permute(2, 0, 1)

            # Action: reuse trainer.collect_rollouts logic
            if closed_loop_policy in ['informative', 'maxdyn']:
                # trainer._init_cem_mu_sig()
                # mu = trainer.init_control.clone()
                # sigma = trainer.sigma.clone()
                mu, costs, sigma = trainer._sample_cem(frame_buffer[-past_len:], mu=mu, sigma=sigma) # pred_len, act_size
                np.savetxt(f'/home/ayush/Desktop/tutorials/rl_latent/E2C/src/data_gen/temp_figs/step_costs.txt', costs.cpu().numpy())
                action_seq = mu.clone()
                # print(mu[0].detach().cpu().numpy())
                plan_obj_vals.append(costs[0].clone().cpu().item())
            else:
                # repeat pred len number of times for action horizon
                act = [env.action_space.sample() for _ in range(pred_len)]
                # print(act[0])
                action_seq = torch.from_numpy(np.array(act)).to(device)
                plan_obj_vals.append(0.0)   # no cost info for random policy
            env_act = action_seq.cpu().detach().numpy()[0]

            # Step env
            for _ in range(trainer.meta_ts):
                obs, rew, done, trunc, _ = env.step(env_act)
                if trunc:
                    print("forced to reset", step_idx)
                    _, _ = env.reset(seed=env_reset_seed + step_idx + 1)
            contact = int(is_robot_contact_geometry(mj_data, robot_geom, obj_geom))
            saved_state[step_idx] = torch.as_tensor([*obs[0:7], rew, *env_act, contact], device='cpu')
            env_rew.append(rew)
            next_render_raw = env.render()
            next_img_true = process_image(next_render_raw, self.dataset_name, downscale=False).permute(2, 0, 1)

            # Model inputs
            with torch.no_grad():
                window = torch.stack(frame_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                mu_prior, log_var_prior, zs = trainer.model.encode_posterior(window)
                h = torch.zeros(model.num_layers, 1, model.deterministic_size, device=trainer.device)
                z = zs[:, -1]
                if trainer.model.output_uncertainty:
                    x_recon, x_pred_uncertainty = trainer.model.decoder(z)
                else:
                    x_recon = trainer.model.decoder(z)
                x_recon_next = []
                for act in action_seq:
                    act_batch = act.view(1, -1).to(trainer.device)
                    h, z_prior, mu_p, log_var_p = trainer.model.rssm_step(h, z.unsqueeze(1), act_batch)
                    if trainer.model.output_uncertainty:
                        x_pred, x_pred_uncertainty = trainer.model.decoder(z_prior)
                    else:
                        x_pred = trainer.model.decoder(z_prior)
                    x_recon_next.append(x_pred.detach().cpu())
                    
                    if trainer.past_length > 1:
                        window_frames = window[:, 1:]   # drop first frame
                        window = torch.cat([window_frames, x_pred.unsqueeze(1).detach()], dim=1)
                    else:
                        window = x_pred.detach()  # past_length==1, just use pred image
                    mu_q, log_var_q, zs = trainer.model.encode_posterior(window)
                    z = zs[:, -1]

            # Feed next frame based on loop type
            next_for_buffer = process_image(next_render_raw, self.dataset_name, downscale=True).permute(2, 0, 1) if closed_loop else x_recon_next[-1].detach().cpu()

            # Update buffers/logs
            frame_buffer.append(next_for_buffer)

            true_frames.append(curr_img)
            recon_frames.append(x_recon.detach().cpu())
            pred_sequences.append(x_recon_next)

        # Build visualization grid: 2 rows, (pred_len + 1) columns
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # ax.set_title(f"Pred t=0; {plan_obj_vals[0]:.2f}")

        ims = []
        # Initialize cells
        ims.clear()
        ims.append(ax.imshow(np.zeros_like(true_frames[0].permute(1, 2, 0))))
        ax.axis('off')

        def update(frame_idx):
            true_curr = true_frames[frame_idx].squeeze(0)
            true_next = true_frames[frame_idx + 1].squeeze(0) if frame_idx + 1 < len(true_frames) else true_curr
            recon = recon_frames[frame_idx].squeeze(0)

            # Pred current recon
            ax.set_title(f"t={frame_idx}; {env_rew[frame_idx]:.2f}")
            # ax[1, 0].set_title(f"True t=0; {env_rew[frame_idx]:.2f}")
            ims[0].set_data(true_curr[:3].permute(1, 2, 0).detach().cpu().numpy())

        ani = FuncAnimation(fig, update, frames=len(true_frames), interval=5.)
        writer = FFMpegWriter(fps=20)
        vid_name = f'{trainer.env_name}_{env_reset_seed}.mp4'
        try:
            filepath = run_path / vid_name
            print(f'Saved planner visualization to {filepath}')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('Exception occurred, saved planner visualization to current directory')
            ani.save(vid_name, writer=writer)
        plt.close(fig)
        return saved_state

2. Create specific test cases with predefined control sequences to evaluate whether the frame skipping is really working. For example, a control sequence of [0., 0., 0.11, 0.] over 5 timesteps should move the end effector 0.055m in +z if dt is 0.11s, by 0.2m if dt is 0.2s, and 0.275m if dt is 0.5m. Show this result as both a plot of the x-y position of the end effector in spatial coordinates with a color gradient for the z dimension, as well as an animation showing the movement. The title of the plot should have the end effector position at a given moment in time.