
const SPEED = 300;
const DISPLAY_WIDTH = 1200;
const DISPLAY_HEIGHT = 600;
const BALL_INIT_VELOCITY_X = 360;
const BALL_INIT_VELOCITY_Y = 0;
const TERMINATE_SCORE = 3;
const PLAYER1_NAME = "WuKong";
const PLAYER2_NAME = "BaJie";


/**
 * The boot scene, entry point of this game
 */
class BootScene extends Phaser.Scene {
    constructor() {
        super({ key: 'BootScene' });
    }

    /**
     * This method is called by the Scene Manager when the scene starts, before preload() and create()
     */
    init() {
    }

    /**
     * Use it to load assets. This method is called by the Scene Manager, after init() and before create()
     */
    preload() {
        this.loadText = this.add.text(DISPLAY_WIDTH * 0.5, DISPLAY_HEIGHT * 0.5, 'Loading ...', { fontFamily: 'Arial', fontSize: 74, color: '#e3f2ed' });
        this.loadText.setOrigin(0.5);
        this.loadText.setStroke('#203c5b', 6);
        this.loadText.setShadow(2, 2, '#2d2d2d', 4, true, false);

        this.tipsText = this.add.text(DISPLAY_WIDTH * 0.5, DISPLAY_HEIGHT * 0.7, '⬆ or W to move up\n⬇ or S to move down', { fontFamily: 'Sans-serif', fontSize: 24, color: '#e3f2ed', lineSpacing: 8 });
        this.tipsText.setOrigin(0.5);

        this.load.spritesheet('ball', 'resources/images/balls.png', { frameWidth: 17, frameHeight: 17 });
        this.load.image('paddle', 'resources/images/paddle.png');
        this.load.image('volume_on', 'resources/images/volume_on.png');
        this.load.image('volume_off', 'resources/images/volume_off.png');
        this.load.image('paused', 'resources/images/paused.png');
        this.load.image('resume', 'resources/images/resume.png');


        this.load.audio('ping_music', 'resources/sound/ping.wav')
        this.load.audio('pong_music', 'resources/sound/pong.wav');
        this.load.audio('wall_music', 'resources/sound/wall.wav')
        this.load.audio('score_music_1', 'resources/sound/score.wav')
        this.load.audio('score_music_2', 'resources/sound/score.mp3')
        this.load.audio("victory", 'resources/sound/victory.mp3')
    }

    /**
     *  Use it to create your game objects. This method is called by the Scene Manager when the scene starts, after init() and preload()
     */
    create() {
        //A global value to hold the terminate score for this game
        this.registry.set('terminate_score', TERMINATE_SCORE);

        this.loadText.setText('Click to Start');
        this.input.on('pointerdown', () => {
            this.scene.transition({
                target: 'PongScene',
                duration: 1000,
                moveBelow: true,
                onUpdate: (progress) => {
                    this.cameras.main.setAlpha(1 - progress);
                }
            });
        });
    }
}

/**
 * The game over scene
 */
class GameOverScene extends Phaser.Scene {
    constructor() {
        super({ key: 'GameOverScene' });
    }

    /**
     * This method is called by the Scene Manager when the scene starts, before preload() and create()
     */
    init() {

    }

    /**
     * Use it to load assets. This method is called by the Scene Manager, after init() and before create()
     */
    preload() {

    }

    /**
     *  Use it to create your game objects. This method is called by the Scene Manager when the scene starts, after init() and preload()
     */
    create() {
        this.victoryAudio = this.sound.add('victory');
        this.victoryAudio.play();

        //  Get the current winner from the registry
        const winner = this.registry.get('winner');
        const textStyle = { fontFamily: 'Arial Black', fontSize: 64, color: '#ffffff', stroke: '#000000', strokeThickness: 8 };

        this.add.text(512, 300, `Game Over\n\Winner:   ${winner}`, textStyle).setAlign('center').setOrigin(0.5);

        this.input.on('pointerdown', () => {
            this.scene.start('BootScene');
        });
    }
}

/**
 * The Pong game logic
 */
class PongScene extends Phaser.Scene {
    constructor() {
        super({ key: 'PongScene' });
    }

    /**
     * This method is called by the Scene Manager when the scene starts, before preload() and create()
     */
    init() {
        this.leftScore = 0;
        this.rightScore = 0;
        this.paused = false;

        this._createVolumeButton();
        this._createPausedButton();

        //the ONNX runtime session
        this.ortSession = null;
        this.leftPaddle = null;
        this.rightPaddle = null;
        this.ball = null;
    }

    /**
     * Use it to load assets. This method is called by the Scene Manager, after init() and before create()
     */
    preload() {
        ort.InferenceSession.create('resources/weight/ppo_model.onnx').then(session => {
            this.ortSession = session
        })
    }

    /**
     *  Use it to create your game objects. This method is called by the Scene Manager when the scene starts, after init() and preload()
     */
    create() {
        // Step 1. add the graphic objects
        const graphics = this.add.graphics();

        // add the game rectangle area
        // args: lineWidth, color, alpha
        graphics.lineStyle(4, 0xffffff, 1.0);

        // draw the middle separate bar between the two players
        const lineLen = DISPLAY_HEIGHT * 0.8 / 20;
        for (let i = 0; i < 20; i++) {
            // args: x1, y1, x2, y2
            graphics.lineBetween(DISPLAY_WIDTH / 2, DISPLAY_HEIGHT * 0.1 + i * lineLen, DISPLAY_WIDTH / 2, DISPLAY_HEIGHT * 0.1 + (i + 0.5) * lineLen);
        }

        // draw the game area bounds rectangle, args: x, y, width, height
        graphics.strokeRect(DISPLAY_WIDTH * 0.1, DISPLAY_HEIGHT * 0.1, DISPLAY_WIDTH * 0.8, DISPLAY_HEIGHT * 0.8);

        // add the glow effect to the graphics. args: color, outerStrength, innerStrength, knockout, quality, distance
        graphics.postFX.addGlow(0x04009E, 5, 0, false, 0.1, 20);

        // Step 2. add the physical objects

        // set the physical bounds. args: x, y, width, height
        this.physics.world.setBounds(DISPLAY_WIDTH * 0.1, DISPLAY_HEIGHT * 0.1, DISPLAY_WIDTH * 0.8, DISPLAY_HEIGHT * 0.8);
        // enable world bounds, but disable the left wall and right wall, args: left, right, up, down
        this.physics.world.setBoundsCollision(false, false, true, true);

        //the left paddle, args: x, y, key
        this.leftPaddle = this.physics.add.image(DISPLAY_WIDTH * 0.12, DISPLAY_HEIGHT * 0.5, 'paddle');
        this.leftPaddle.setImmovable(true);
        //sets whether this body collides with the world boundary, args: value, bounceX, bounceY
        this.leftPaddle.setCollideWorldBounds(true, 1, 1);
        //sets the bounce values of this body, args: x, y
        this.leftPaddle.setBounce(1, 1);
        this.leftPaddle.name = "left_paddle";
        //x, y, width, height
        this.leftPaddle.body.setBoundsRectangle(new Phaser.Geom.Rectangle(DISPLAY_WIDTH * 0.1, DISPLAY_HEIGHT * 0.1, DISPLAY_WIDTH * 0.8, DISPLAY_HEIGHT * 0.8));

        //the right paddle, args: x, y, key
        this.rightPaddle = this.physics.add.image(DISPLAY_WIDTH * 0.88, DISPLAY_HEIGHT * 0.5, 'paddle')
        this.rightPaddle.setImmovable(true);
        //sets whether this body collides with the world boundary, args: value, bounceX, bounceY
        this.rightPaddle.setCollideWorldBounds(true, 1, 1);
        //sets the bounce values of this body, args: x, y
        this.rightPaddle.setBounce(1, 1);
        this.rightPaddle.name = "right_paddle";
        //x, y, width, height
        this.rightPaddle.body.setBoundsRectangle(new Phaser.Geom.Rectangle(DISPLAY_WIDTH * 0.1, DISPLAY_HEIGHT * 0.1, DISPLAY_WIDTH * 0.8, DISPLAY_HEIGHT * 0.8));

        //add ball to physics, args: x, y, key, frame
        const ball = this.physics.add.sprite(DISPLAY_WIDTH * 0.5, DISPLAY_HEIGHT * 0.5, 'ball', 3)
            .setCollideWorldBounds(true, 1, 1)
            .setBounce(1);
        ball.body.onWorldBounds = true;
        this.ball = ball;

        //add the collider for ball and the paddles
        this.physics.add.collider(ball, [this.leftPaddle, this.rightPaddle], this.onPaddle, null, this);

        // Step 3. add the music
        this.pingAudio = this.sound.add('ping_music');
        this.pongAudio = this.sound.add('pong_music');
        this.wallAudio = this.sound.add('wall_music');
        this.score1Audio = this.sound.add('score_music_1');
        this.score2Audio = this.sound.add('score_music_2');

        // add the ball&world bounds collider music
        this.physics.world.on("worldbounds", (body, blockedUp, blockedDown, blockedLeft, blockedRight) => {
            if (body.gameObject == this.ball) {
                this.wallAudio.play();
            }
        })

        // Step 4. add the score board. x, y, text, font-attr
        this.scoreBoard = this.add.text(DISPLAY_WIDTH * 0.5, DISPLAY_HEIGHT * 0.05, '0:0', { fontFamily: 'monospace', fontSize: 45, color: '#00ff00', stroke: '#000000', strokeThickness: 3 })
            .setAlign('center').setOrigin(0.5, 0.5);

        // Step 5. add the players name
        this.player1Name = this.add.text(DISPLAY_WIDTH * 0.1, DISPLAY_HEIGHT * 0.08, PLAYER1_NAME, { fontFamily: 'Arial Black', fontSize: 20, color: '#00ff00', stroke: '#000000', strokeThickness: 1 })
            .setAlign('left').setOrigin(0.0, 1.0);

        this.player2Name = this.add.text(DISPLAY_WIDTH * 0.9, DISPLAY_HEIGHT * 0.08, PLAYER2_NAME, { fontFamily: 'Arial Black', fontSize: 20, color: '#00ff00', stroke: '#000000', strokeThickness: 1 })
            .setAlign('right').setOrigin(1.0, 1.0);

        // Step 6. start to server the ball
        this.time.addEvent({
            delay: 3000,  //ms
            callback: () => { ball.body.velocity.set(BALL_INIT_VELOCITY_X * Phaser.Math.RND.sign(), BALL_INIT_VELOCITY_Y + Phaser.Math.Between(-480, 480)); },
            //args: [],
            callbackScope: this,
            repeat: 0,
            loop: false
        });
    }

    update(time, delta) {
        if (this.ball.x > DISPLAY_WIDTH) {
            //left win
            this._onBallOut("left")
        } else if (this.ball.x < 0) {
            //right win
            this._onBallOut("right")
        }

        if (this.paused) {
            return;
        }

        //move the left&right paddle
        this._movePaddles()
    }

    onPaddle(ball, paddle) {
        //play the sound
        if (paddle.name == "left_paddle") {
            this.pingAudio.play();
        } else {
            this.pongAudio.play();
        }

        //change the ball y velocity
        ball.body.velocity.y = Math.random() * 50 + paddle.body.velocity.y;
    }

    _onBallOut(winner) {
        if (winner == "right") {
            //the right player scores
            this.rightScore = this.rightScore + 1;
        } else if (winner == "left") {
            //the left player scores
            this.leftScore = this.leftScore + 1;
        }

        //the terminate score
        let terminateScore = this.registry.get("terminate_score");
        if (this.leftScore >= terminateScore) {
            this.registry.set("winner", PLAYER1_NAME);
            this.scene.start('GameOverScene');
            return;
        } else if (this.rightScore >= terminateScore) {
            this.registry.set("winner", PLAYER2_NAME);
            this.scene.start('GameOverScene');
            return;
        }

        if (winner == "right") {
            //the right player scores
            //play the sound
            this.score1Audio.play();
        } else if (winner == "left") {
            //the left player scores
            //play the sound
            this.score2Audio.play();
        }

        //update the score board
        this.scoreBoard.setText(this.leftScore + ":" + this.rightScore);

        this.ball.body.reset(DISPLAY_WIDTH * 0.5, DISPLAY_HEIGHT * 0.5)
        this.time.addEvent({
            delay: 3000,  //ms
            callback: () => { this.ball.body.velocity.set(BALL_INIT_VELOCITY_X * Phaser.Math.RND.sign(), BALL_INIT_VELOCITY_Y + Phaser.Math.Between(-480, 480)); },
            //args: [],
            callbackScope: this,
            repeat: 0,
            loop: false
        });
    }

    _movePaddles() {
        //move the paddle by AI agent
        if (this.ortSession) {
            const right_x = (this.ball.x - DISPLAY_WIDTH * 0.1) / (DISPLAY_WIDTH * 0.8)
            const right_y = 1.0 - (this.ball.y - DISPLAY_HEIGHT * 0.1) / (DISPLAY_HEIGHT * 0.8)

            const right_velocity_x = 0.2 * this.ball.body.velocity.x / (DISPLAY_WIDTH * 0.8)
            const right_velocity_y = 0.2 * (-this.ball.body.velocity.y / (DISPLAY_HEIGHT * 0.8))

            const right_paddle_y = 1 - (this.rightPaddle.y - DISPLAY_HEIGHT * 0.1) / (DISPLAY_HEIGHT * 0.8)

            const left_x = 1 - right_x
            const left_y = right_y
            const left_velocity_x = -right_velocity_x
            const left_velocity_y = right_velocity_y
            const left_paddle_y = 1 - (this.leftPaddle.y - DISPLAY_HEIGHT * 0.1) / (DISPLAY_HEIGHT * 0.8)

            const gameState = Float32Array.from([right_x, right_y, right_velocity_x, right_velocity_y, right_paddle_y, left_x, left_y, left_velocity_x, left_velocity_y, left_paddle_y])
            const tensorInput = new ort.Tensor('float32', gameState, [2, 5])
            this.ortSession.run({ input: tensorInput }).then(result => {
                if (result.output.data[0] == 0) {
                    //right paddle action: move up
                    this.rightPaddle.body.velocity.y = -1 * SPEED;
                } else {
                    //right paddle action: move down
                    this.rightPaddle.body.velocity.y = SPEED;
                }

                if (result.output.data[1] == 0) {
                    //left paddle action: move up
                    this.leftPaddle.body.velocity.y = -1 * SPEED;
                } else {
                    //left paddle action: move down
                    this.leftPaddle.body.velocity.y = SPEED;
                }
            })
        }
    }

    _createVolumeButton() {
        const volumeButton = this.add.image(40, 40, "volume_on").setName("volume_on");
        volumeButton.setInteractive();

        // Mouse enter
        volumeButton.on(Phaser.Input.Events.POINTER_OVER, () => {
            this.input.setDefaultCursor("pointer");
        });
        // Mouse leave
        volumeButton.on(Phaser.Input.Events.POINTER_OUT, () => {
            this.input.setDefaultCursor("default");
        });

        volumeButton.on(Phaser.Input.Events.POINTER_DOWN, () => {
            if (this.sound.volume === 0) {
                this.sound.setVolume(1);
                volumeButton.setTexture("volume_on");
                volumeButton.setAlpha(1);
            } else {
                this.sound.setVolume(0);
                volumeButton.setTexture("volume_off");
                volumeButton.setAlpha(.5)
            }
        });
    }

    _createPausedButton() {
        const pausedButton = this.add.image(DISPLAY_WIDTH - 40, 40, "paused").setName("paused");
        pausedButton.setInteractive();

        // Mouse enter
        pausedButton.on(Phaser.Input.Events.POINTER_OVER, () => {
            this.input.setDefaultCursor("pointer");
        });
        // Mouse leave
        pausedButton.on(Phaser.Input.Events.POINTER_OUT, () => {
            this.input.setDefaultCursor("default");
        });

        pausedButton.on(Phaser.Input.Events.POINTER_DOWN, () => {
            this.paused = !this.paused;

            if (this.paused) {
                pausedButton.setTexture("resume");
                pausedButton.setAlpha(1);

                this.pingAudio.stop();
                this.pongAudio.stop();
                this.wallAudio.stop();
                this.score1Audio.stop();
                this.score2Audio.stop();
                this.physics.pause();
            } else {
                pausedButton.setTexture("paused");
                pausedButton.setAlpha(.5)

                this.physics.resume();
            }
        });
    }
}

//the game config: https://docs.phaser.io/api-documentation/typedef/types-core
const config = {
    type: Phaser.WEBGL,
    width: DISPLAY_WIDTH,
    height: DISPLAY_HEIGHT,
    parent: "game_area",
    backgroundColor: '#26062B',
    scale: {
        mode: Phaser.Scale.NONE,
        autoCenter: Phaser.Scale.CENTER_BOTH
    },
    physics: {
        default: 'arcade',
        arcade: {
            debug: false
        }
    },

    scene: [BootScene, PongScene, GameOverScene],
};

//register window onload event listener
window.onload = () => {
    console.log("window.onload, the game starts here.........")

    //the Phaser game object
    const game = new Phaser.Game(config);
};