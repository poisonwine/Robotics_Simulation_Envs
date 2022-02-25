# TODO: add the following commented environments into the register
# from BaxterReacherv0 import *
# from myenvs.robosuite.robosuite import *

import copy
from .registration import register, make, registry, spec


register(
    id='FetchThrowDice-v0',
    entry_point='myenvs.fetch:FetchThrowDiceEnv',
    kwargs={},
    max_episode_steps=50,
)

register(
    id='Catcher3d-v0',
    entry_point='myenvs.fetch.moving_target.catch:Catcher3dEnv',
    max_episode_steps=50,
    kwargs={},
 )

register(
    id='DyReach-v0',
    entry_point='myenvs.fetch.moving_target.dyreach:DyReachEnv',
    max_episode_steps=50,
    kwargs={'direction': (1, 0, 0),
            'velocity': 0.011}
)

register(
    id='DyCircle-v0',
    entry_point='myenvs.fetch.moving_target.dycircle:DyCircleEnv',
    max_episode_steps=50,
    kwargs={'velocity': 0.05,
            'center_offset': [1.3, 0.7, 0]}
)
register(
    id='DyPush-v0',
    entry_point='myenvs.fetch.moving_target.dypush:DyPushEnv',
    max_episode_steps=50,
    kwargs={
        'velocity': 0.011
})



    register(
        id='FetchReachDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchReachDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPushDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchSlideDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchSlideDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrowRubberBall{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowRubberBallEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPickAndThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='BaxterPickAndPlace{}-v0'.format(suffix),
        entry_point='myenvs.baxter.pick_and_place:BaxterPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterPickAndPlace{}-v2'.format(suffix),
        entry_point='myenvs.baxter.pick_and_place_obstacle:BaxterPickAndPlaceObstacleEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='BaxterSlide{}-v0'.format(suffix),
        entry_point='myenvs.baxter.slide:BaxterSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterPush{}-v0'.format(suffix),
        entry_point='myenvs.baxter.push:BaxterPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterPushObstacle{}-v1'.format(suffix),
        entry_point='myenvs.baxter.push_obstacle:BaxterPushObstacleEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='BaxterKitting{}-v0'.format(suffix),
        entry_point='myenvs.baxter.kitting:BaxterKittingEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterReach{}-v0'.format(suffix),
        entry_point='myenvs.baxter.reach:BaxterReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterGolf{}-v0'.format(suffix),
        entry_point='myenvs.baxter.golf:BaxterGolfEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )



    register(
        id='FetchPickAndPlace{}-v2'.format(suffix),
        entry_point='myenvs.fetch.pick_and_place_hard:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushWallObstacle{}-v1'.format(suffix),
        entry_point = 'myenvs.fetch.push_wall_obstacle:FetchPushWallObstacle_v1',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPushWallObstacle{}-v2'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v2',
        kwargs=kwargs,
        max_episode_steps=100,
    )
    register(
        id='FetchPushWallObstacle{}-v3'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v3',
        kwargs=kwargs,
        max_episode_steps=100,
    )
    register(
        id='FetchPnPObstacle{}-v1'.format(suffix),
        entry_point='myenvs.fetch.pick_and_place_obstacle:FetchPnPObstacle_v1',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPushWallObstacle{}-v4'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v4',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPushWallObstacle{}-v5'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v5',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='DragRope{}-v0'.format(suffix),
        entry_point='myenvs.ravens:DragRopeEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='SweepPile{}-v0'.format(suffix),
        entry_point='myenvs.ravens:SweepPileEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )


    _rope_kwargs = {
        'observation_mode': 'key_point',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 50,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1,
        'use_cached_states': False,
        'deterministic': False,
        'save_cached_states': False,
    }
    _rope_kwargs.update(kwargs)
    register(
        id='RopeConfiguration{}-v0'.format(suffix),
        entry_point='myenvs.softgymenvs:RopeConfigurationEnv',
        kwargs=_rope_kwargs,
        max_episode_steps=50,
    )
