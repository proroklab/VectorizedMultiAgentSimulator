import torch
import numpy as np

from maps.simulator.core import Agent, World, Landmark, Sphere, Box, Line
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color, X, Y


class Scenario(BaseScenario):


    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world)
        self.init_ball(world)
        self.init_background(world)
        self.init_walls(world)
        self.init_goals(world)
        self.init_traj_pts(world)
        return world


    def reset_world_at(self, env_index: int = None):
        self.reset_ball(env_index)
        self.reset_agents(env_index)
        self.reset_background(env_index)
        self.reset_walls(env_index)
        self.reset_goals(env_index)
        self.reset_controllers(env_index)


    def init_params(self, **kwargs):
        self.viewer_size = kwargs.get("viewer_size", (1200, 800))
        self.ai_red_agents = kwargs.get("ai_red_agents", True)
        self.ai_blue_agents = kwargs.get("ai_blue_agents", False)
        self.n_blue_agents = kwargs.get("n_blue_agents", 3)
        self.n_red_agents = kwargs.get("n_red_agents", 3)
        self.agent_size = kwargs.get("agent_size", 0.025)
        self.goal_size = kwargs.get("goal_size", 0.35)
        self.goal_depth = kwargs.get("goal_depth", 0.1)
        self.pitch_length = kwargs.get("pitch_length", 3.0)
        self.pitch_width = kwargs.get("pitch_width", 1.5)
        self.max_speed = kwargs.get("max_speed", 0.15)
        self.u_multiplier = kwargs.get("u_multiplier", 0.1)
        self.ball_max_speed = kwargs.get("ball_max_speed", 0.3)
        self.ball_mass = kwargs.get("ball_mass", 0.1)
        self.ball_size = kwargs.get("ball_size", 0.02)
        self.n_traj_points = kwargs.get("n_traj_points", 8)


    def init_world(self, batch_dim: int, device: torch.device):
        # Make world
        world = World(
            batch_dim,
            device,
            dt=0.1,
            damping=0.05,
            contact_force=1e2,
            x_semidim=self.pitch_length / 2 + self.goal_depth - self.agent_size,
            y_semidim=self.pitch_width / 2 - self.agent_size,
        )
        world.agent_size = self.agent_size
        world.pitch_width = self.pitch_width
        world.pitch_length = self.pitch_length
        world.goal_size = self.goal_size
        world.goal_depth = self.goal_depth
        return world


    def init_agents(self, world):
        # Add agents
        self.blue_controller = None
        self.red_controller = None
        if self.ai_blue_agents:
            self.blue_controller = AgentPolicy()
        if self.ai_red_agents:
            self.red_controller = AgentPolicy()

        blue_agents = []
        for i in range(self.n_blue_agents):
            agent = Agent(name=f"Agent Blue {i}",
                          shape=Sphere(radius=self.agent_size),
                          action_script=self.blue_controller.policy if self.ai_blue_agents else None,
                          u_multiplier=self.u_multiplier,
                          max_speed=self.max_speed,
                          color=Color.BLUE)
            world.add_agent(agent)
            blue_agents.append(agent)

        red_agents = []
        for i in range(self.n_red_agents):
            agent = Agent(name=f"Agent Red {i}",
                          shape=Sphere(radius=self.agent_size),
                          action_script=self.red_controller.policy if self.ai_red_agents else None,
                          u_multiplier=self.u_multiplier,
                          max_speed=self.max_speed,
                          color=Color.RED)
            world.add_agent(agent)
            red_agents.append(agent)

        self.red_agents = red_agents
        self.blue_agents = blue_agents
        world.red_agents = red_agents
        world.blue_agents = blue_agents


    def reset_agents(self, env_index: int = None):
        for agent in self.blue_agents:
            agent.set_pos(
                torch.rand(
                    self.world.dim_p
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device
                )
                + torch.tensor([-self.pitch_length / 2, -self.pitch_width / 2]),
                batch_index=env_index,
            )
            agent.set_vel(
                torch.zeros(2, device=self.world.device),
                batch_index=env_index,
            )
        for agent in self.red_agents:
            agent.set_pos(
                torch.rand(
                    self.world.dim_p
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device
                )
                + torch.tensor([0., -self.pitch_width / 2]),
                batch_index=env_index,
            )
            agent.set_vel(
                torch.zeros(2, device=self.world.device),
                batch_index=env_index,
            )


    def reset_controllers(self, env_index: int = None):
        if self.red_controller is not None:
            if not self.red_controller.initialised:
                self.red_controller.init(self.world)
            self.red_controller.reset(env_index)
        if self.blue_controller is not None:
            if not self.blue_controller.initialised:
                self.blue_controller.init(self.world)
            self.blue_controller.reset(env_index)


    def init_ball(self, world):
        # Add Ball
        ball = Agent(name=f"Ball",
                     shape=Sphere(radius=self.ball_size),
                     action_script=ball_action_script,
                     max_speed=self.ball_max_speed,
                     mass=self.ball_mass,
                     color=Color.GRAY)
        world.add_agent(ball)
        world.ball = ball
        self.ball = ball


    def reset_ball(self, env_index: int = None):
        self.ball.set_pos(
            torch.zeros(2, device=self.world.device),
            batch_index=env_index,
        )
        self.ball.set_vel(
            torch.zeros(2, device=self.world.device),
            batch_index=env_index,
        )


    def init_background(self, world):
        # Add landmarks
        background = Landmark(
            name=f"Background",
            collide=False,
            movable=False,
            shape=Box(length=self.pitch_length, width=self.pitch_width),
            color=Color.GREEN,
        )
        world.add_landmark(background)

        centre_circle_outer = Landmark(
            name=f"Centre Circle Outer",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.goal_size / 2),
            color=Color.WHITE,
        )
        world.add_landmark(centre_circle_outer)

        centre_circle_inner = Landmark(
            name=f"Centre Circle Inner",
            collide=False,
            movable=False,
            shape=Sphere(self.goal_size / 2 - 0.02),
            color=Color.GREEN,
        )
        world.add_landmark(centre_circle_inner)

        centre_line = Landmark(
            name=f"Centre Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2 * self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(centre_line)

        right_line = Landmark(
            name=f"Right Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2 * self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(right_line)

        left_line = Landmark(
            name=f"Left Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2 * self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(left_line)

        top_line = Landmark(
            name=f"Top Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_length - 2 * self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(top_line)

        bottom_line = Landmark(
            name=f"Bottom Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_length - 2 * self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(bottom_line)


    def reset_background(self, env_index: int = None):
        for landmark in self.world.landmarks:
            if landmark.name == "Centre Line":
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Line":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2 - self.agent_size, 0.], dtype=torch.float32, device=self.world.device,),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Line":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2 + self.agent_size, 0.], dtype=torch.float32, device=self.world.device,),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Top Line":
                landmark.set_pos(
                    torch.tensor([0., self.pitch_width/2 - self.agent_size], dtype=torch.float32, device=self.world.device,),
                    batch_index=env_index,
                )
            elif landmark.name == "Bottom Line":
                landmark.set_pos(
                    torch.tensor([0., -self.pitch_width/2 + self.agent_size], dtype=torch.float32, device=self.world.device,),
                    batch_index=env_index,
                )


    def init_walls(self, world):

        right_top_wall = Landmark(
            name=f"Right Top Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.pitch_width/2 - self.agent_size - self.goal_size/2, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(right_top_wall)

        left_top_wall = Landmark(
            name=f"Left Top Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.pitch_width/2 - self.agent_size - self.goal_size/2, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(left_top_wall)

        right_bottom_wall = Landmark(
            name=f"Right Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.pitch_width/2 - self.agent_size - self.goal_size/2, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(right_bottom_wall)

        left_bottom_wall = Landmark(
            name=f"Left Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(length=self.pitch_width/2 - self.agent_size - self.goal_size/2, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(left_bottom_wall)


    def reset_walls(self, env_index: int = None):
        for landmark in self.world.landmarks:
            if landmark.name == "Left Top Wall":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length / 2, self.pitch_width / 4 + self.goal_size / 4], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )

            elif landmark.name == "Left Bottom Wall":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length / 2, -self.pitch_width / 4 - self.goal_size / 4], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )

            elif landmark.name == "Right Top Wall":
                landmark.set_pos(
                    torch.tensor([self.pitch_length / 2, self.pitch_width / 4 + self.goal_size / 4], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Bottom Wall":
                landmark.set_pos(
                    torch.tensor([self.pitch_length / 2, -self.pitch_width / 4 - self.goal_size / 4], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )


    def init_goals(self, world):
        right_goal_back = Landmark(
            name=f"Right Goal Back",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(right_goal_back)

        left_goal_back = Landmark(
            name=f"Left Goal Back",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(left_goal_back)

        right_goal_top = Landmark(
            name=f"Right Goal Top",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(right_goal_top)

        left_goal_top = Landmark(
            name=f"Left Goal Top",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(left_goal_top)

        right_goal_bottom = Landmark(
            name=f"Right Goal Bottom",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(right_goal_bottom)

        left_goal_bottom = Landmark(
            name=f"Left Goal Bottom",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(left_goal_bottom)

        blue_net = Landmark(
            name=f"Blue Net",
            collide=False,
            movable=False,
            shape=Box(length=self.goal_depth, width=self.goal_size),
            color=(0.5, 0.5, 0.5, 0.5),
        )
        world.add_landmark(blue_net)

        red_net = Landmark(
            name=f"Red Net",
            collide=False,
            movable=False,
            shape=Box(length=self.goal_depth, width=self.goal_size),
            color=(0.5, 0.5, 0.5, 0.5),
        )
        world.add_landmark(red_net)

        self.blue_net = blue_net
        self.red_net = red_net
        world.blue_net = blue_net
        world.red_net = red_net


    def reset_goals(self, env_index: int = None):
        for landmark in self.world.landmarks:
            if landmark.name == "Left Goal Back":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length / 2 - self.goal_depth + self.agent_size, 0.], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Back":
                landmark.set_pos(
                    torch.tensor([self.pitch_length / 2 + self.goal_depth - self.agent_size, 0.], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Goal Top":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length / 2 - self.goal_depth / 2 + self.agent_size, self.goal_size / 2],
                                 dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Goal Bottom":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length / 2 - self.goal_depth / 2 + self.agent_size, -self.goal_size / 2],
                                 dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Top":
                landmark.set_pos(
                    torch.tensor([self.pitch_length / 2 + self.goal_depth / 2 - self.agent_size, self.goal_size / 2],
                                 dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Bottom":
                landmark.set_pos(
                    torch.tensor([self.pitch_length / 2 + self.goal_depth / 2 - self.agent_size, -self.goal_size / 2],
                                 dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Red Net":
                landmark.set_pos(
                    torch.tensor([self.pitch_length / 2 + self.goal_depth / 2 - self.agent_size / 2, 0.], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Blue Net":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length / 2 - self.goal_depth / 2 + self.agent_size / 2, 0.], dtype=torch.float32,
                                 device=self.world.device, ),
                    batch_index=env_index,
                )


    def init_traj_pts(self, world):
        world.traj_points = {"Red": {}, "Blue": {}}
        if self.ai_red_agents:
            for i, agent in enumerate(world.red_agents):
                world.traj_points["Red"][agent] = []
                for j in range(self.n_traj_points):
                    pointj = Landmark(
                        name="Red {agent} Trajectory {pt}".format(agent=i, pt=j),
                        collide=False,
                        movable=False,
                        shape=Sphere(radius=0.01),
                        color=Color.GRAY,
                    )
                    world.add_landmark(pointj)
                    world.traj_points["Red"][agent].append(pointj)
        if self.ai_blue_agents:
            for i, agent in enumerate(world.blue_agents):
                world.traj_points["Blue"][agent] = []
                for j in range(self.n_traj_points):
                    pointj = Landmark(
                        name="Blue {agent} Trajectory {pt}".format(agent=i, pt=j),
                        collide=False,
                        movable=False,
                        shape=Sphere(radius=0.01),
                        color=Color.GRAY,
                    )
                    world.add_landmark(pointj)
                    world.traj_points["Blue"][agent].append(pointj)


    def reward(self, agent: Agent):
        if agent == self.world.agents[0]:
            over_right_line = self.ball.state.pos[:,0] > self.pitch_length / 2 + self.ball_size / 2
            in_right_goal = self.world.is_overlapping(self.ball, self.red_net)
            over_left_line = self.ball.state.pos[:, 0] < -self.pitch_length / 2 - self.ball_size / 2
            in_left_goal = self.world.is_overlapping(self.ball, self.blue_net)
            right_goal = over_right_line & in_right_goal
            left_goal = over_left_line & in_left_goal
            self._reward = 1 * right_goal - 1 * left_goal
            self._done = right_goal | left_goal
        return self._reward


    def observation(self, agent: Agent):
        obs =  torch.cat(
            [agent.state.pos,
             agent.state.vel,
             self.ball.state.pos - agent.state.pos,
             self.ball.state.vel - agent.state.vel],
            dim=1
        )
        return obs


    def done(self):
        return self._done


### Ball Physics ###


def ball_action_script(ball, world):
    # Avoid getting stuck against the wall
    dist_thres = world.agent_size * 2
    vel_thres = 0.1
    impulse = 0.01
    upper = 1 - torch.minimum(world.pitch_width / 2 - ball.state.pos[:,1], torch.tensor(dist_thres)) / dist_thres
    lower = 1 - torch.minimum(world.pitch_width / 2 + ball.state.pos[:,1], torch.tensor(dist_thres)) / dist_thres
    right = 1 - torch.minimum(world.pitch_length / 2 - ball.state.pos[:, 0], torch.tensor(dist_thres)) / dist_thres
    left = 1 - torch.minimum(world.pitch_length / 2 + ball.state.pos[:, 0], torch.tensor(dist_thres)) / dist_thres
    vertical_vel = 1 - torch.minimum(torch.abs(ball.state.vel[:,1]), torch.tensor(vel_thres)) / vel_thres
    horizontal_vel = 1 - torch.minimum(torch.abs(ball.state.vel[:, 1]), torch.tensor(vel_thres)) / vel_thres
    dist_action = torch.stack([left-right, lower-upper], dim=1)
    vel_action = torch.stack([horizontal_vel, vertical_vel], dim=1)
    actions = dist_action * vel_action * impulse
    goal_mask = (ball.state.pos[:, 1] < world.goal_size/2) * (ball.state.pos[:, 1] > -world.goal_size/2)
    actions[goal_mask,0] = 0
    ball.action.u = actions


### Agent Policy ###


class AgentPolicy:

    def __init__(self, team="Red"):
        self.team_name = team
        self.otherteam_name = "Blue" if (self.team_name == "Red") else "Red"

        self.pos_lookahead = 0.01
        self.vel_lookahead = 0.01
        self.start_vel_mag = 0.6

        self.dribble_speed = 0.5
        self.dribble_slowdown_dist = 0.25
        self.dribble_stop_margin_vel_coeff = 0.1
        self.initial_vel_dist_behind_target_frac = 0.3
        self.ball_pos_eps = 0.08

        self.max_shoot_dist = 0.6
        self.valid_start_cos = np.cos(np.pi/8)
        self.valid_start_dist = 0.2
        self.dist_to_hit_speed = 1.7
        self.start_vel_mag_shoot = 1.0
        self.touch_eps = 0.01

        self.initialised = False


    def init(self, world):
        self.initialised = True
        self.world = world

        self.ball = self.world.ball
        if self.team_name == "Red":
            self.teammates = self.world.red_agents
            self.opposition = self.world.blue_agents
            self.own_net = self.world.red_net
            self.target_net = self.world.blue_net
        elif self.team_name == "Blue":
            self.teammates = self.world.blue_agents
            self.opposition = self.world.red_agents
            self.own_net = self.world.blue_net
            self.target_net = self.world.red_net

        self.actions = {
            agent:
                {
                    "moving": torch.zeros(self.world.batch_dim).bool(),
                    "dribbling": torch.zeros(self.world.batch_dim).bool(),
                    "shooting": torch.zeros(self.world.batch_dim).bool(),
                    "pre-shooting": torch.zeros(self.world.batch_dim).bool(),
                }
            for agent in self.teammates
        }

        self.action_steps = {
            agent: torch.zeros(self.world.batch_dim).int() for agent in self.teammates
        }

        self.objectives = {
            agent:
                {
                    "target_pos": torch.zeros(self.world.batch_dim, self.world.dim_p),
                    "target_vel": torch.zeros(self.world.batch_dim, self.world.dim_p),
                    "start_pos": torch.zeros(self.world.batch_dim, self.world.dim_p),
                    "start_vel": torch.zeros(self.world.batch_dim, self.world.dim_p),
                    "norm_dist": torch.zeros(self.world.batch_dim),
                }
            for agent in self.teammates
        }


    def reset(self, env_index = slice(None)):
        for agent in self.teammates:
            self.actions[agent]["moving"][env_index] = False
            self.actions[agent]["dribbling"][env_index] = False
            self.actions[agent]["shooting"][env_index] = False
            self.actions[agent]["pre-shooting"][env_index] = False
            self.action_steps[agent][env_index] = 0
            self.objectives[agent]["target_pos"][env_index] = torch.zeros(self.world.dim_p)
            self.objectives[agent]["target_vel"][env_index] = torch.zeros(self.world.dim_p)
            self.objectives[agent]["start_pos"][env_index] = torch.zeros(self.world.dim_p)
            self.objectives[agent]["start_vel"][env_index] = torch.zeros(self.world.dim_p)


    def policy(self, agent, world):
        # self.dribble(agent, self.target_net.state.pos)
        self.dribble(agent, torch.tensor([-0.75, 0.]).unsqueeze(0), env_index=slice(None))
        # self.go_to(agent, torch.tensor([0., 0.]).unsqueeze(0), torch.tensor([0., 1.]).unsqueeze(0))
        # self.shoot(agent, torch.tensor([-0.6, 0.]).unsqueeze(0), env_index=[0])
        # self.shoot(agent, self.target_net.state.pos, env_index=[0])
        control = self.get_action(agent)
        control = torch.clamp(control, min=-1., max=1.)
        agent.action.u = control * agent.u_multiplier


    def shoot(self, agent, pos, env_index=slice(None)):
        if isinstance(env_index, int):
            env_index = [env_index]

        ball_curr_pos = self.ball.state.pos[env_index]
        agent_curr_pos = agent.state.pos[env_index]

        ball_target_disp = pos - ball_curr_pos
        ball_target_dist = ball_target_disp.norm(dim=-1)
        ball_target_dir = ball_target_disp / ball_target_dist

        agent_ball_disp = ball_curr_pos - agent_curr_pos
        agent_ball_dist = agent_ball_disp.norm(dim=-1)
        agent_ball_dir = agent_ball_disp / agent_ball_dist

        dist_maxdist_ratio = (torch.minimum(ball_target_dist, torch.tensor(self.max_shoot_dist)) / self.max_shoot_dist)

        # Determine if shooting or pre-shooting
        start_dist = self.valid_start_dist * dist_maxdist_ratio
        valid_angle_mask = (ball_target_dir * agent_ball_dir).sum(dim=-1) > self.valid_start_cos
        valid_dist_mask = agent_ball_dist > start_dist
        shooting_mask = self.actions[agent]["shooting"][env_index] | (valid_dist_mask & valid_angle_mask)
        pre_shooting_mask = ~shooting_mask
        start_shooting_mask = ~self.actions[agent]["shooting"][env_index] & shooting_mask
        start_pre_shooting_mask = ~self.actions[agent]["pre-shooting"][env_index] & pre_shooting_mask
        self.action_steps[agent][self.combine_mask(env_index, start_shooting_mask & start_pre_shooting_mask)] = 0
        self.actions[agent]["shooting"][env_index] = shooting_mask
        self.actions[agent]["pre-shooting"][env_index] = pre_shooting_mask

        # Shooting
        hit_pos = ball_curr_pos - ball_target_dir * (self.ball.shape.radius + agent.shape.radius)
        hit_speed = self.dist_to_hit_speed * dist_maxdist_ratio
        hit_vel = ball_target_dir * hit_speed
        start_vel = self.get_start_vel(hit_pos, hit_vel, agent_curr_pos, hit_speed)

        # Pre Shooting
        pre_shoot_target_pos = ball_curr_pos - ball_target_dir * start_dist
        pre_shoot_target_vel = ball_target_dir * hit_speed

        # Next to wall
        close_to_wall_mask = self.clamp_pos(pre_shoot_target_pos, return_bool=True) & pre_shooting_mask
        pre_shooting_mask = pre_shooting_mask & ~close_to_wall_mask
        self.update_dribble(
            agent,
            pos=pos[close_to_wall_mask],
            env_index=self.combine_mask(env_index, close_to_wall_mask),
        )

        self.go_to(
            agent,
            pos=pre_shoot_target_pos[pre_shooting_mask],
            vel=pre_shoot_target_vel[pre_shooting_mask],
            env_index=self.combine_mask(env_index, pre_shooting_mask),
        )

        self.go_to(
            agent,
            pos=hit_pos[shooting_mask],
            vel=hit_vel[shooting_mask],
            start_vel=start_vel,
            env_index=self.actions[agent]["shooting"],
        )

        touch_dist = (ball_curr_pos - agent_curr_pos).norm(dim=-1) - (self.ball.shape.radius + agent.shape.radius)
        touch_mask = touch_dist < self.touch_eps
        self.actions[agent]["shooting"][self.combine_mask(env_index, shooting_mask)] = ~touch_mask


    def dribble(self, agent, pos, env_index=slice(None)):
        if isinstance(env_index, int):
            env_index = [env_index]

        self.action_steps[agent][self.combine_mask(env_index, ~self.actions[agent]["dribbling"][env_index])] = 0
        self.actions[agent]["dribbling"][env_index] = True

        dist = (pos - self.ball.state.pos[env_index]).norm(dim=-1)
        reached_goal_mask = self.combine_mask(env_index, dist <= self.ball_pos_eps)
        self.actions[agent]["dribbling"][reached_goal_mask] = False
        self.action_steps[agent][reached_goal_mask] = 0
        curr_pos = agent.state.pos[reached_goal_mask]
        self.go_to(agent, curr_pos, torch.zeros(curr_pos.shape), env_index=~self.actions[agent]["dribbling"])

        self.update_dribble(agent, pos, env_index=self.actions[agent]["dribbling"])
        self.action_steps[agent][self.actions[agent]["dribbling"]] += 1


    def update_dribble(self, agent, pos, env_index=slice(None)):
        agent_pos = agent.state.pos[env_index]
        ball_pos = self.ball.state.pos[env_index]
        ball_disp = pos - ball_pos
        ball_dist = ball_disp.norm(dim=-1)
        direction = ball_disp / ball_dist[:,None]
        hit_pos = ball_pos - direction * (self.ball.shape.radius + agent.shape.radius)
        hit_vel = direction * self.dribble_speed
        start_vel = self.get_start_vel(hit_pos, hit_vel, agent_pos, self.start_vel_mag)

        slowdown_mask = ball_dist <= self.dribble_slowdown_dist
        hit_vel[slowdown_mask,:] *= ball_dist[slowdown_mask,None] / self.dribble_slowdown_dist
        # start_vel[slowdown_mask,:] *= ball_dist[slowdown_mask,None] / self.dribble_slowdown_dist

        self.go_to(agent, hit_pos, hit_vel, start_vel=start_vel, env_index=env_index)


    def go_to(self, agent, pos, vel, start_vel=None, norm_dist=0, env_index=slice(None)):
        start_pos = agent.state.pos[env_index]
        if start_vel is None:
            start_vel = self.get_start_vel(pos, vel, start_pos, self.start_vel_mag)
        self.objectives[agent]["target_pos"][env_index] = pos
        self.objectives[agent]["target_vel"][env_index] = vel
        self.objectives[agent]["start_pos"][env_index] = start_pos
        self.objectives[agent]["start_vel"][env_index] = start_vel
        self.objectives[agent]["norm_dist"][env_index] = norm_dist
        self.plot_traj(agent, env_index=env_index)


    def get_start_vel(self, pos, vel, start_pos, start_vel_mag):
        goal_disp = pos - start_pos
        goal_dist = goal_disp.norm(dim=-1)
        vel_mag = vel.norm(dim=-1)
        vel_dir = vel.clone()
        vel_dir[vel_mag > 0] /= vel_mag[vel_mag > 0, None]
        dist_behind_target = self.initial_vel_dist_behind_target_frac * goal_dist
        target_pos = pos - vel_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = target_disp.norm(dim=1)
        start_vel_aug_dir = target_disp
        start_vel_aug_dir[target_dist > 0] /= target_dist[target_dist > 0, None]
        start_vel = start_vel_aug_dir * start_vel_mag
        return start_vel


    def get_action(self, agent, env_index=slice(None)):
        curr_pos = agent.state.pos[env_index, :]
        curr_vel = agent.state.vel[env_index, :]
        u_start = torch.zeros(curr_pos.shape[0])
        des_curr_pos = self.hermite(
            self.objectives[agent]["start_pos"][env_index, :],
            self.objectives[agent]["target_pos"][env_index, :],
            self.objectives[agent]["start_vel"][env_index, :],
            self.objectives[agent]["target_vel"][env_index, :],
            u = np.minimum(u_start + self.pos_lookahead, 1.),
            deriv = 0,
        )
        des_curr_vel = self.hermite(
            self.objectives[agent]["start_pos"][env_index, :],
            self.objectives[agent]["target_pos"][env_index, :],
            self.objectives[agent]["start_vel"][env_index, :],
            self.objectives[agent]["target_vel"][env_index, :],
            u = np.minimum(u_start + self.vel_lookahead, 1.),
            deriv = 1,
        )
        des_curr_pos = torch.as_tensor(des_curr_pos)
        des_curr_vel = torch.as_tensor(des_curr_vel)
        control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (des_curr_vel - curr_vel)
        return control


    def hermite(self, p0, p1, p0dot, p1dot, u=0.1, deriv=0, norm_dist=None):
        # Formatting
        u = self.to_numpy(u)
        p0 = self.to_numpy(p0)
        p1 = self.to_numpy(p1)
        p0dot = self.to_numpy(p0dot)
        p1dot = self.to_numpy(p1dot)
        input_shape = p0.shape
        p0 = p0.reshape(-1, p0.shape[-1])
        p1 = p1.reshape(-1, p1.shape[-1])
        p0dot = p0dot.reshape(-1, p0dot.shape[-1])
        p1dot = p1dot.reshape(-1, p1dot.shape[-1])
        u = u.reshape((-1,))

        if norm_dist is not None:
            norm_dist = self.to_numpy(norm_dist).reshape(-1)
            norm_dist_mask = norm_dist > 0
            p0p1_disp = (p1 - p0)
            p0p1_dist = np.linalg.norm(p0p1_disp, axis=-1)
            p0p1_dist[p0p1_dist == 0] = 1
            p1[norm_dist_mask] = p0[norm_dist_mask] + p0p1_disp[norm_dist_mask] / p0p1_dist[norm_dist_mask,None] * norm_dist[norm_dist_mask,None]

        # Calculation
        U = np.array([self.nPr(3,deriv) * (u ** max(0,3-deriv)),
                      self.nPr(2,deriv) * (u ** max(0,2-deriv)),
                      self.nPr(1,deriv) * (u ** max(0,1-deriv)),
                      self.nPr(0,deriv) * (u ** 0)], dtype=np.float64).T
        A = np.array([[2. ,-2., 1., 1.],
                      [-3., 3.,-2.,-1.],
                      [0. , 0., 1., 0.],
                      [1. , 0., 0., 0.]])
        P = np.stack([p0, p1, p0dot, p1dot], axis=1)
        ans = U[:,None,:] @ A[None,:,:] @ P
        ans = ans.squeeze(1)
        if (norm_dist is not None) and deriv == 0:
            norm_dist_mask = norm_dist > 0
            p0_in = p0.reshape(*input_shape)
            p0_in = p0_in.reshape(-1, p0_in.shape[-1])
            ans[norm_dist_mask] = (ans - p0_in)[norm_dist_mask] * p0p1_dist[norm_dist_mask,None] / norm_dist[norm_dist_mask, None] + p0_in[norm_dist_mask]
        ans = ans.reshape(*input_shape)
        return ans


    def invhermite(self, pos, p0, p1, p0dot, p1dot):
        input_shape = pos.shape
        pos = self.to_numpy(pos)
        p0 = self.to_numpy(p0)
        p1 = self.to_numpy(p1)
        p0dot = self.to_numpy(p0dot)
        p1dot = self.to_numpy(p1dot)


        A = np.array([[2. ,-2., 1., 1.],
                      [-3., 3.,-2.,-1.],
                      [0. , 0., 1., 0.],
                      [1. , 0., 0., 0.]])
        P = np.array([p0.reshape((-1,)), p1.reshape((-1,)), p0dot.reshape((-1,)), p1dot.reshape((-1,))])
        mat1 = np.matmul(A, P)
        mat1[3,:] -= pos.reshape((-1,))

        # Dot product
        mat1 = mat1.reshape(4, *input_shape)
        if len(mat1.shape) == 2:
            mat1 = mat1[:,None,:]
        polys = []
        for i in range(mat1.shape[1]):
            polysi = []
            for j in range(mat1.shape[2]):
                polyij = np.polymul(np.poly1d(mat1[:,i,j]), np.poly1d(mat1[:,i,j]))
                polysi.append(polyij)
            polyi = sum(polysi)
            polys.append(polyi)

        ubest = np.zeros(len(polys))
        for i in range(len(polys)):
            # Find roots
            polyroots = np.roots(polys[i])

            # Filter into range [0,1]. Add endpoints to check.
            mask = (polyroots > 0) & (polyroots < 1)
            uposs = np.real(polyroots[mask])
            uposs = np.concatenate([np.array([0,1]), uposs])

            # Choose best root
            bestdist = float('inf')
            ubesti = 0
            for j in range(len(uposs)):
                phat = self.hermite(p0[i,:], p1[i,:], p0dot[i,:], p1dot[i,:], uposs[j])
                dist = np.linalg.norm(pos - phat)
                if dist < bestdist:
                    bestdist = dist
                    ubesti = uposs[j]
            ubest[i] = ubesti
        return ubest


    def plot_traj(self, agent, env_index=0):
        for i, u in enumerate(np.linspace(0,1,len(self.world.traj_points[self.team_name][agent]))):
            pointi = self.world.traj_points[self.team_name][agent][i]
            num_envs = self.objectives[agent]["start_pos"][env_index, :].shape[0]
            posi = self.hermite(
                self.objectives[agent]["start_pos"][env_index, :],
                self.objectives[agent]["target_pos"][env_index, :],
                self.objectives[agent]["start_vel"][env_index, :],
                self.objectives[agent]["target_vel"][env_index, :],
                norm_dist=self.objectives[agent]["norm_dist"][env_index],
                u=torch.tensor([u] * num_envs),
                deriv=0,
            )
            if env_index==slice(None) or (isinstance(env_index, torch.Tensor) and env_index.dtype == torch.bool and torch.all(env_index)):
                pointi.set_pos(torch.as_tensor(posi), batch_index=None)
            elif isinstance(env_index, int):
                pointi.set_pos(torch.as_tensor(posi), batch_index=env_index)
            elif isinstance(env_index, list):
                for envi in env_index:
                    pointi.set_pos(torch.as_tensor(posi)[envi,:], batch_index=env_index[envi])
            elif isinstance(env_index, torch.Tensor) and env_index.dtype == torch.bool and torch.any(env_index):
                envs = torch.where(env_index)
                for i, envi in enumerate(envs):
                    pointi.set_pos(torch.as_tensor(posi)[i,:], batch_index=envi[0])


    def clamp_pos(self, pos, return_bool=False):
        orig_pos = pos.clone()
        agent_size = self.world.agent_size
        pitch_y = self.world.pitch_width / 2 - agent_size
        pitch_x = self.world.pitch_length / 2 - agent_size
        goal_y = self.world.goal_size / 2 - agent_size
        goal_x = self.world.goal_depth
        pos[:,Y] = torch.clamp(pos[:,Y], -pitch_y, pitch_y)
        inside_goal_y_mask = torch.abs(pos[:,Y]) < goal_y
        pos[~inside_goal_y_mask, X] = torch.clamp(pos[~inside_goal_y_mask, X], -pitch_x, pitch_x)
        pos[inside_goal_y_mask, X] = torch.clamp(pos[inside_goal_y_mask, X], -pitch_x-goal_x, pitch_x+goal_x)
        if return_bool:
            return torch.any(pos != orig_pos, dim=-1)
        else:
            return pos


    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.numpy()
        elif isinstance(x, list):
            return np.array(x)
        return np.array(x)


    def nPr(self, n, r):
        if r > n:
            return 0
        ans = 1
        for k in range(n,max(1,n-r),-1):
            ans = ans * k
        return ans


    def combine_mask(self, env_index, mask):
        if env_index == slice(None):
            return mask
        elif isinstance(env_index, torch.Tensor) and env_index.dtype == torch.bool:
            return torch.arange(env_index.shape[0])[mask]
        elif isinstance(env_index, torch.Tensor) and env_index.dtype == torch.int:
            return env_index[mask]
        elif isinstance(env_index, list):
            return torch.tensor(env_index)[mask]


### Run ###

def interactive():
    from maps.interactive_rendering import render_interactively
    render_interactively(
        "football",
        continuous=True,
        n_blue_agents=1,
        n_red_agents=1,
    )


def multiple_envs():
    from PIL import Image
    from maps import make_env

    scenario_name = "football"

    # Scenario specific variables
    env_args = {
        "n_blue_agents": 1,
        "n_red_agents": 2,
    }

    num_envs = 6
    render_env = 2
    continuous_actions = False
    device = "cpu"  # or cuda or any other torch device
    n_steps = 10000

    action = [0., 0.]

    env = make_env(
        scenario_name=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        rllib_wrapped=False,
        # Environment specific variables
        **env_args,
    )

    step = 0
    n_agents = env_args["n_blue_agents"]
    for s in range(n_steps):
        actions = []
        step += 1
        for i in range(n_agents):
            actions.append(
                torch.tensor(
                    action,
                    device=device,
                ).repeat(num_envs, 1)
            )
        obs, rews, dones, info = env.step(actions)
        env.render(
            mode="rgb_array",
            agent_index_focus=None,
            visualize_when_rgb=True,
            env_index=render_env,
        )


if __name__ == '__main__':
    multiple_envs()
