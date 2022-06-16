import torch
import numpy as np

from maps.simulator.core import Agent, World, Landmark, Sphere, Box, Line
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
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
        self.max_speed = kwargs.get("max_speed", 0.1)
        self.u_multiplier = kwargs.get("u_multiplier", 0.1)
        self.ball_max_speed = kwargs.get("ball_max_speed", 0.2)
        self.ball_mass = kwargs.get("ball_mass", 0.1)
        self.ball_size = kwargs.get("ball_size", 0.02)
        self.n_traj_points = kwargs.get("n_traj_points", 8)

        self.hits = []
        self.last_hit_obs = None
        self.ball_hit = False

        # Make world
        world = World(
            batch_dim,
            device,
            dt=0.1,
            damping=0.05,
            contact_force=5e1,
            x_semidim=self.pitch_length/2 + self.goal_depth - self.agent_size,
            y_semidim=self.pitch_width/2 - self.agent_size,
        )

        # Add agents
        self.blue_controller = None
        self.red_controller = None
        if self.ai_blue_agents:
            self.blue_controller = AgentPolicy()
        if self.ai_red_agents:
            self.red_controller = AgentPolicy()
        world.blue_agents = []
        for i in range(self.n_blue_agents):
            agent = Agent(name=f"Agent Blue {i}",
                          shape=Sphere(radius=self.agent_size),
                          action_script=self.blue_controller.policy if self.ai_blue_agents else None,
                          u_multiplier=self.u_multiplier,
                          max_speed=self.max_speed,
                          color=Color.BLUE)
            world.add_agent(agent)
            world.blue_agents.append(agent)
        world.red_agents = []
        for i in range(self.n_red_agents):
            agent = Agent(name=f"Agent Red {i}",
                          shape=Sphere(radius=self.agent_size),
                          action_script=self.red_controller.policy if self.ai_red_agents else None,
                          u_multiplier=self.u_multiplier,
                          max_speed=self.max_speed,
                          color=Color.RED)
            world.add_agent(agent)
            world.red_agents.append(agent)

        # Add Ball
        ball = Agent(name=f"Ball",
                     shape=Sphere(radius=self.ball_size),
                     action_script=ball_action_script,
                     max_speed=self.ball_max_speed,
                     mass = self.ball_mass,
                     color=Color.GRAY)
        world.add_agent(ball)
        world.agent_size = self.agent_size
        world.pitch_width = self.pitch_width
        world.pitch_length = self.pitch_length
        world.goal_size = self.goal_size

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
            shape=Sphere(radius=self.goal_size/2),
            color=Color.WHITE,
        )
        world.add_landmark(centre_circle_outer)

        centre_circle_inner = Landmark(
            name=f"Centre Circle Inner",
            collide=False,
            movable=False,
            shape=Sphere(self.goal_size/2-0.02),
            color=Color.GREEN,
        )
        world.add_landmark(centre_circle_inner)

        centre_line = Landmark(
            name=f"Centre Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2*self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(centre_line)

        right_line = Landmark(
            name=f"Right Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2*self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(right_line)

        left_line = Landmark(
            name=f"Left Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2*self.agent_size, width=6),
            color=Color.WHITE,
        )
        world.add_landmark(left_line)

        top_line = Landmark(
            name=f"Top Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_length - 2*self.agent_size, width=6),
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

        self.ball = ball
        self.left_net = blue_net
        self.right_net = red_net
        return world

    def reset_world_at(self, env_index: int = None):
        self.ball.set_pos(
            torch.zeros(2, device=self.world.device),
            batch_index=env_index,
        )
        self.ball.set_vel(
            torch.zeros(2, device=self.world.device),
            batch_index=env_index,
        )
        for i, agent in enumerate(self.world.agents):
            if "Blue" in agent.name:
                agent.set_pos(
                    torch.rand(
                        self.world.dim_p
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p),
                        device=self.world.device
                    )
                    + torch.tensor([-self.pitch_length/2, -self.pitch_width/2]),
                    batch_index=env_index,
                )
            elif "Red" in agent.name:
                agent.set_pos(
                    torch.rand(
                        self.world.dim_p
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p),
                        device=self.world.device
                    )
                    + torch.tensor([0., -self.pitch_width/2]),
                    batch_index=env_index,
                )
            agent.set_vel(
                torch.zeros(2, device=self.world.device),
                batch_index=env_index,
            )
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
            elif landmark.name == "Left Top Wall":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2, self.pitch_width/4 + self.goal_size/4], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Bottom Wall":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2, -self.pitch_width/4 - self.goal_size/4], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Top Wall":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2, self.pitch_width/4 + self.goal_size/4], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Bottom Wall":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2, -self.pitch_width/4 - self.goal_size/4], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Goal Back":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2 - self.goal_depth + self.agent_size, 0.], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Back":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2 + self.goal_depth - self.agent_size, 0.], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Goal Top":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2 - self.goal_depth/2 + self.agent_size, self.goal_size/2], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Goal Bottom":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2 - self.goal_depth/2  + self.agent_size, -self.goal_size/2], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Top":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2 + self.goal_depth/2 - self.agent_size, self.goal_size/2], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Bottom":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2 + self.goal_depth/2 - self.agent_size, -self.goal_size/2], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            elif landmark.name == "Red Net":
                landmark.set_pos(
                    torch.tensor([self.pitch_length/2 + self.goal_depth/2 - self.agent_size/2, 0.], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
                landmark.color
            elif landmark.name == "Blue Net":
                landmark.set_pos(
                    torch.tensor([-self.pitch_length/2 - self.goal_depth/2 + self.agent_size/2, 0.], dtype=torch.float32, device=self.world.device, ),
                    batch_index=env_index,
                )
            if self.red_controller is not None:
                self.red_controller.init(self.world)
            if self.blue_controller is not None:
                self.blue_controller.init(self.world)

    def reward(self, agent: Agent):
        if agent == self.world.agents[0]:
            over_right_line = self.ball.state.pos[:,0] > self.pitch_length / 2 + self.ball_size / 2
            in_right_goal = self.world.is_overlapping(self.ball, self.right_net)
            over_left_line = self.ball.state.pos[:, 0] < -self.pitch_length / 2 - self.ball_size / 2
            in_left_goal = self.world.is_overlapping(self.ball, self.left_net)
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
             self.ball.state.vel - agent.state.vel]
        )
        return obs

    def done(self):
        return self._done



## Ball Physics ##

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
    return ball.action



## Agent Policy ##

class AgentPolicy:

    def __init__(self, team="Red"):
        self.team_name = team
        self.otherteam_name = "Blue" if (team=="Red") else "Red"
        self.lookahead = 0.01
        self.dribble_speed = 0.5
        self.touch_eps = 0.02
        self.vel_eps = 0.01
        self.min_action_steps = 1
        self.max_action_steps = 100
        self.objectives = {}
        self.actions = {}
        self.action_steps = {}
        self.teammates = []
        self.opposition = []
        self.initialised = False

    def init(self, world):
        self.initialised = True
        self.world = world

        for agent in world.agents:
            if agent.name == "Ball":
                self.ball = agent
            elif self.team_name in agent.name:
                self.teammates.append(agent)
            elif self.otherteam_name in agent.name:
                self.opposition.append(agent)

        for landmark in world.landmarks:
            if "Net" in landmark.name:
                if self.team_name in landmark.name:
                    self.own_net = landmark
                elif self.otherteam_name in landmark.name:
                    self.target_net = landmark

        self.actions = {
            agent: {"moving": False, "dribbling": False, "passing": False, "shooting": False} for agent in self.teammates
        }

        self.action_steps = {
            agent: 0 for agent in self.teammates
        }

        for agent in self.teammates:
            self.go_to(agent, agent.state.pos, agent.state.vel)


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


    def hermite(self, p0, p1, p0dot, p1dot, u=0.1, deriv=0, norm_dist=None):
        # Formatting
        u = self.to_numpy(u)
        p0 = self.to_numpy(p0)
        p1 = self.to_numpy(p1)
        p0dot = self.to_numpy(p0dot)
        p1dot = self.to_numpy(p1dot)
        input_shape = p0.shape

        if norm_dist is not None:
            p0p1_disp = p1 - p0
            p0p1_dist = np.linalg.norm(p0p1_disp, axis=-1)
            if p0p1_dist == 0:
                p0p1_dist = 1
            p1 = p0 + p0p1_disp / p0p1_dist * norm_dist

        u = u.reshape((-1,))
        p0 = p0.reshape((-1,))
        p1 = p1.reshape((-1,))
        p0dot = p0dot.reshape((-1,))
        p1dot = p1dot.reshape((-1,))
        # Calculation
        U = np.array([self.nPr(3,deriv) * (u ** max(0,3-deriv)),
                      self.nPr(2,deriv) * (u ** max(0,2-deriv)),
                      self.nPr(1,deriv) * (u ** max(0,1-deriv)),
                      self.nPr(0,deriv) * (u ** 0)], dtype=np.float64).T
        A = np.array([[2. ,-2., 1., 1.],
                      [-3., 3.,-2.,-1.],
                      [0. , 0., 1., 0.],
                      [1. , 0., 0., 0.]])
        P = np.array([p0,p1,p0dot,p1dot],dtype=np.float64)
        ans = U.dot(A).dot(P)
        ans = ans.reshape(*input_shape)
        if (norm_dist is not None) and deriv == 0:
            ans = (ans - p0) * p0p1_dist / norm_dist + p0
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
                polyij = np.polymul(np.poly1d(mat1[:,i,j]),np.poly1d(mat1[:,i,j]))
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


    def go_to(self, agent, pos, vel, start_vel=None):
        start_pos = agent.state.pos.clone()
        if start_vel is None:
            goal_disp = pos - start_pos
            goal_dist = goal_disp.norm(dim=1)
            start_vel_aug_dir = goal_disp / goal_dist if (goal_dist > 0) else goal_disp
            start_vel = start_vel_aug_dir * vel.norm(dim=1)
        self.objectives[agent] = {
            "target_pos": pos.numpy(),
            "target_vel": vel.numpy(),
            "start_pos": start_pos.numpy(),
            "start_vel": start_vel.numpy(),
        }
        self.plot_traj(agent)


    def plot_traj(self, agent):
        for i, u in enumerate(np.linspace(0,1,len(self.world.traj_points[self.team_name][agent]))):
            pointi = self.world.traj_points[self.team_name][agent][i]
            posi = self.hermite(
                self.objectives[agent]["start_pos"],
                self.objectives[agent]["target_pos"],
                self.objectives[agent]["start_vel"],
                self.objectives[agent]["target_vel"],
                u=u,
                deriv=0,
            )
            pointi.set_pos(torch.as_tensor(posi), batch_index=0)


    def get_action(self, agent):
        curr_pos = agent.state.pos
        u_closest = self.invhermite(
            curr_pos,
            self.objectives[agent]["start_pos"],
            self.objectives[agent]["target_pos"],
            self.objectives[agent]["start_vel"],
            self.objectives[agent]["target_vel"]
        )
        des_curr_pos = self.hermite(
            self.objectives[agent]["start_pos"],
            self.objectives[agent]["target_pos"],
            self.objectives[agent]["start_vel"],
            self.objectives[agent]["target_vel"],
            u = np.minimum(u_closest + self.lookahead, 1.),
            deriv = 0,
        )
        des_curr_vel = self.hermite(
            self.objectives[agent]["start_pos"],
            self.objectives[agent]["target_pos"],
            self.objectives[agent]["start_vel"],
            self.objectives[agent]["target_vel"],
            u = np.minimum(u_closest + self.lookahead, 1.),
            deriv = 1,
        )
        des_curr_pos = torch.as_tensor(des_curr_pos)
        des_curr_vel = torch.as_tensor(des_curr_vel)
        control = 0.5 * (des_curr_pos - agent.state.pos) + 0.5 * (des_curr_vel - agent.state.vel)
        return control


    def update_dribble(self, agent, pos):
        ball_pos = self.ball.state.pos
        direction = pos - ball_pos
        direction = direction / direction.norm(dim=1)
        hit_pos = ball_pos - direction * (self.ball.shape.radius + agent.shape.radius)
        hit_vel = direction * self.dribble_speed
        self.go_to(agent, hit_pos, hit_vel)


    def dribble_stop_cond(self, agent):
        if self.action_steps[agent] < self.min_action_steps:
            return False
        if self.action_steps[agent] >= self.max_action_steps:
            return True
        disp = self.ball.state.pos - agent.state.pos
        dist = disp.norm(dim=1)
        if dist < self.ball.shape.radius + agent.shape.radius + self.touch_eps:
            return True
        return False

    def dribble(self, agent, pos):
        if self.actions[agent]["dribbling"]:
            if self.dribble_stop_cond(agent):
                self.actions[agent]["dribbling"] = False
                self.action_steps[agent] = 0
            else:
                self.action_steps[agent] += 1
        if not self.actions[agent]["dribbling"]:
            self.actions[agent]["dribbling"] = True
            self.update_dribble(agent, pos)


    def policy(self, agent, world):
        if not self.initialised:
            self.init(agent, world)
        self.dribble(agent, self.target_net.state.pos)
        control = self.get_action(agent)
        control = torch.clamp(control, min=-1., max=1.)
        agent.action.u = control * agent.u_multiplier
        return agent.action





## Run ##

if __name__ == '__main__':
    from maps.interactive_rendering import render_interactively
    render_interactively(
        "football",
        continuous=True,
        n_blue_agents=1,
        n_red_agents=1,
    )
