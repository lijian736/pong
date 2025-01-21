from typing import Tuple
import random
import math
from Box2D import (
    b2World,
    b2FixtureDef,
    b2PolygonShape,
    b2CircleShape,
    b2ContactListener,
)


class ObjectsContactListener(b2ContactListener):
    """
    the physical world objects contact listener
    """

    def BeginContact(self, contact):
        user_data1 = contact.fixtureA.body.userData
        user_data2 = contact.fixtureB.body.userData

        if (
            user_data1
            and user_data2
            and user_data1["name"] in ["paddle", "ball"]
            and user_data2["name"] in ["paddle", "ball"]
        ):
            user_data1["collide"] = True
            user_data2["collide"] = True

    def EndContact(self, contact):
        pass


class GameWorld:
    """
    the ball game world

    args:
        area_width: the game area width
        area_height: the game area height
    """

    def __init__(self, area_width: int, area_height: int):
        # the action space
        self.action_space = [0, 1]
        self.action_meaning = {0: "UP", 1: "DOWN"}

        # the area definition
        self.area_width = area_width
        self.area_height = area_height

        # create the physical world
        self.physical_world = b2World(gravity=(0, 0))
        # the collide listener
        self.physical_world.contactListener = ObjectsContactListener()

        # the top&bottom&left wall
        self.top_wall = self.physical_world.CreateStaticBody(
            position=(area_width / 2, area_height),
            shapes=b2PolygonShape(box=(area_width / 2 + 4, 1)),
        )
        self.bottom_wall = self.physical_world.CreateStaticBody(
            position=(area_width / 2, 0),
            shapes=b2PolygonShape(box=(area_width / 2 + 4, 1)),
        )
        self.left_wall = self.physical_world.CreateStaticBody(
            position=(0, area_height / 2),
            shapes=b2PolygonShape(box=(1, area_height / 2 + 4)),
        )

        # the paddle x default position
        self.paddle_x = self.area_width - 10

    @property
    def height(self):
        return self.area_height

    @property
    def width(self):
        return self.area_width

    @property
    def actions(self):
        return self.action_space

    def reset(self) -> Tuple[float, float, float, float, float]:
        """
        reset the game

        returns: the game default state
        """
        # create the paddle with a circle shape
        self.paddle = self.physical_world.CreateDynamicBody(
            position=(self.paddle_x, self.area_height / 2)
        )
        self.paddle.CreateFixture(
            shape=b2CircleShape(pos=(0, 0), radius=0.1),
            density=1,
            friction=0,
            restitution=1,
        )
        self.paddle.userData = {"name": "paddle", "collide": False}

        # create the ball
        self.ball = self.physical_world.CreateDynamicBody(
            position=(
                10,
                random.uniform(10, self.area_height - 10),
            )
        )

        self.ball.CreateFixture(
            shape=b2CircleShape(pos=(0, 0), radius=0.1),
            density=1,
            friction=0,
            restitution=1,
        )
        self.ball.userData = {"name": "ball", "collide": False}

        # https://stackoverflow.com/questions/14774202/is-there-an-upper-limit-on-velocity-when-using-box2d
        # There is a maximum movement limit of 2.0 units per time step, given in the file b2Settings.h in the source code
        self.ball.linearVelocity.x = 59 * random.uniform(1, 2)
        self.ball.linearVelocity.y = 59 * random.uniform(-2, 2)

        return (
            self.ball.position.x / self.area_width,
            self.ball.position.y / self.area_height,
            self.ball.linearVelocity.x / self.area_width,
            self.ball.linearVelocity.y / self.area_height,
            self.paddle.position.y / self.area_height,
        )

    def destroy_bodies(self):
        self.physical_world.DestroyBody(self.ball)
        self.physical_world.DestroyBody(self.paddle)

    def check_over(self):
        ball_position = self.ball.position
        is_ball_out = ball_position.x >= self.paddle_x

        ball_user_data = self.ball.userData
        paddle_user_data = self.paddle.userData

        is_collide = ball_user_data["collide"] and paddle_user_data["collide"]
        return is_ball_out or is_collide

    def step(self, action: int = 0, step_num: int = 1, paddle_height: int = 50):
        """
        take one step forward
        return: game state, reward, done
        """
        if action == 0:
            # UP
            self.paddle.linearVelocity.y = 110
        elif action == 1:
            # DOWN
            self.paddle.linearVelocity.y = -110
        else:
            self.paddle.linearVelocity.y = 0

        time_step = 1.0 / 60
        velocity_iterations = 20
        position_iterations = 20

        done = False
        for _ in range(step_num):
            self.physical_world.Step(
                time_step, velocity_iterations, position_iterations
            )

            done = self.check_over()
            if done:
                break

        ball_position = self.ball.position
        ball_velocity = self.ball.linearVelocity

        paddle_pos = self.paddle.position.y

        reward = 0
        hit = False

        if done:
            reward = -math.log(
                abs(paddle_pos - ball_position.y) / self.area_height + 0.000001
            )
            hit = abs(paddle_pos - ball_position.y) < (paddle_height / 2)

        return (
            (
                ball_position.x / self.area_width,
                ball_position.y / self.area_height,
                ball_velocity.x / self.area_width,
                ball_velocity.y / self.area_height,
                paddle_pos / self.area_height,
            ),
            reward,
            done,
            hit,
        )
