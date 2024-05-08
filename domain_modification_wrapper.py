"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, CameraModder, DynamicsModder

DEFAULT_COLOR_ARGS = {
    'geom_names' : None, # all geoms are randomized
    'randomize_local' : True, # sample nearby colors
    'randomize_material' : True, # randomize material reflectance / shininess / specular
    'local_rgb_interpolation' : 0.2,
    'local_material_interpolation' : 0.3,
    'texture_variations' : ['rgb', 'checker', 'noise', 'gradient'], # all texture variation types
    'randomize_skybox' : True, # by default, randomize skybox too
}

DEFAULT_CAMERA_ARGS = {
    'camera_names' : None, # all cameras are randomized
    'randomize_position' : True,
    'randomize_rotation' : True,
    'randomize_fovy' : True,
    'position_perturbation_size' : 0.01,
    'rotation_perturbation_size' : 0.087,
    'fovy_perturbation_size' : 5.,
}

DEFAULT_LIGHTING_ARGS = {
    'light_names' : None, # all lights are randomized
    'randomize_position' : True,
    'randomize_direction' : True,
    'randomize_specular' : True,
    'randomize_ambient' : True,
    'randomize_diffuse' : True,
    'randomize_active' : True,
    'position_perturbation_size' : 0.1,
    'direction_perturbation_size' : 0.35,
    'specular_perturbation_size' : 0.1,
    'ambient_perturbation_size' : 0.1,
    'diffuse_perturbation_size' : 0.1,
}

DEFAULT_DYNAMICS_ARGS = {
    # Opt parameters
    'randomize_density': True,
    'randomize_viscosity': True,
    'density_perturbation_ratio': 0.1,
    'viscosity_perturbation_ratio': 0.1,

    # Body parameters
    'body_names': None,     # all bodies randomized
    'randomize_position': True,
    'randomize_quaternion': True,
    'randomize_inertia': True,
    'randomize_mass': True,
    'position_perturbation_size': 0.0015,
    'quaternion_perturbation_size': 0.003,
    'inertia_perturbation_ratio': 0.02,
    'mass_perturbation_ratio': 0.02,

    # Geom parameters
    'geom_names': None,     # all geoms randomized
    'randomize_friction': True,
    'randomize_solref': True,
    'randomize_solimp': True,
    'friction_perturbation_ratio': 0.1,
    'solref_perturbation_ratio': 0.1,
    'solimp_perturbation_ratio': 0.1,

    # Joint parameters
    'joint_names': None,    # all joints randomized
    'randomize_stiffness': True,
    'randomize_frictionloss': True,
    'randomize_damping': True,
    'randomize_armature': True,
    'stiffness_perturbation_ratio': 0.1,
    'frictionloss_perturbation_size': 0.05,
    'damping_perturbation_size': 0.01,
    'armature_perturbation_size': 0.01,
}


class DomainModificationWrapper(Wrapper):
    """
    Wrapper that allows for domain randomization mid-simulation.

    Args:
        env (MujocoEnv): The environment to wrap.

        seed (int): Integer used to seed all randomizations from this wrapper. It is
            used to create a np.random.RandomState instance to make sure samples here
            are isolated from sampling occurring elsewhere in the code. If not provided,
            will default to using global random state.

        randomize_color (bool): if True, randomize geom colors and texture colors

        randomize_camera (bool): if True, randomize camera locations and parameters

        randomize_lighting (bool): if True, randomize light locations and properties

        randomize_dyanmics (bool): if True, randomize dynamics parameters

        color_randomization_args (dict): Color-specific randomization arguments

        camera_randomization_args (dict): Camera-specific randomization arguments

        lighting_randomization_args (dict): Lighting-specific randomization arguments

        dynamics_randomization_args (dict): Dyanmics-specific randomization arguments

        randomize_on_reset (bool): if True, randomize on every call to @reset. This, in
            conjunction with setting @randomize_every_n_steps to 0, is useful to
            generate a new domain per episode.

        randomize_every_n_steps (int): determines how often randomization should occur. Set
            to 0 if randomization should happen manually (by calling @randomize_domain)

    """
    def __init__(
        self, 
        env,
        modded_geom_names=None,
        modded_geom_rgbs=None,
        seed=None,
        randomize_on_reset=True,
        randomize_every_n_steps=0,
    ):
        super().__init__(env)

        self.seed = seed
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None
        self.randomize_on_reset = randomize_on_reset
        self.randomize_every_n_steps = randomize_every_n_steps
        self.modded_geom_names = modded_geom_names
        self.modded_geom_rgbs = modded_geom_rgbs

        self.step_counter = 0

        

        self.modders = []
        self.tex_modder = TextureModder(sim=self.env.sim)

        for i in range(len(self.modded_geom_names)):
            self.tex_modder.set_rgb(self.modded_geom_names[i], self.modded_geom_rgbs[i])

        # set object colors

        # I think I don't need this line
        # self.tex_modder.whiten_materials()  # ensures materials won't impact colors
        
        # lift   
        # self.tex_modder.set_rgb('cube_g0_vis', (30,144,255))
        # self.tex_modder.set_rgb('table_visual', (160,82,45))

        # can
        # self.tex_modder.set_rgb('floor', (245,245,220))
        # self.tex_modder.set_rgb('Can_g0_visual', (50,205,50))

        # square
        # self.tex_modder.set_rgb('table_visual', (222,184,135))
        # self.tex_modder.set_rgb('SquareNut_g0_visual', (50,205,50))
        # self.tex_modder.set_rgb('SquareNut_g1_visual', (50,205,50))
        # self.tex_modder.set_rgb('SquareNut_g2_visual', (50,205,50))
        # self.tex_modder.set_rgb('SquareNut_g3_visual', (50,205,50))
        # self.tex_modder.set_rgb('SquareNut_g4_visual', (50,205,50))

        # tool_hang
        # self.tex_modder.set_rgb('table_visual', (222,184,135))
        # self.tex_modder.set_rgb('wall_front_visual', (0,0,128))
        # self.tex_modder.set_rgb('tool_handle_g0_vis', (0,0,0))

        # transport
        # self.tex_modder.set_rgb('floor', (245,245,220))
        # self.tex_modder.set_rgb('wall_leftcorner_visual', (240,255,255))
        # self.tex_modder.set_rgb('wall_rightcorner_visual', (240,255,255))
        # self.tex_modder.set_rgb('wall_left_visual', (240,255,255))
        # self.tex_modder.set_rgb('wall_right_visual', (240,255,255))
        # self.tex_modder.set_rgb('wall_rear_visual', (240,255,255))
        # self.tex_modder.set_rgb('wall_front_visual', (240,255,255))
        # self.tex_modder.set_rgb('payload_handle_vis', (0,0,0))
        # self.tex_modder.set_rgb('payload_head_vis', (0,0,0))
        # self.tex_modder.set_rgb('trash_g0_vis', (0,0,0))
        # self.tex_modder.set_rgb('transport_start_bin_lid_handle_vis', (255,20,147))

        self.save_default_domain()

    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # undo all randomizations
        self.restore_default_domain()

        # normal env reset
        ret = super().reset()

        # save the original env parameters
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency
        self.step_counter = 0

        # update sims
        for modder in self.modders:
            modder.update_sim(self.env.sim)

        if self.randomize_on_reset:
            # domain randomize + regenerate observation
            self.randomize_domain()
            ret = self.env._get_observations()

        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain randomization

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        # functionality for randomizing at a particular frequency
        if self.randomize_every_n_steps > 0:
            if self.step_counter % self.randomize_every_n_steps == 0:
                self.randomize_domain()
        self.step_counter += 1

        return super().step(action)

    def randomize_domain(self):
        """
        Runs domain randomization over the environment.
        """
        for modder in self.modders:
            modder.randomize()

    def save_default_domain(self):
        """
        Saves the current simulation model parameters so
        that they can be restored later.
        """
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        Restores the simulation model parameters saved
        in the last call to @save_default_domain.
        """
        for modder in self.modders:
            modder.restore_defaults()
