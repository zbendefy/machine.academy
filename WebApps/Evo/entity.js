class Entity {
    constructor(pixelGetter, neuralNetwork = null) {
        if (neuralNetwork === null)
            this.brain = new NeuralNetwork(GetInitialNetwork());
        else
            this.brain = neuralNetwork;
        this.pixelGetter = pixelGetter;
        this.imgWidth = 512;
        this.imgHeight = 512;
        
        this.maxSeeingDistancePx = 100;

        this.thinkTimeS = 0.05;

        this.carSteeringSpeed = 0.5;
        this.carAcceleration = 8.0;
        this.carBrakeStrength = 2.0;
        this.carAirResistance = 0.01;
        
        this.sensor_spreadView = 0.45;
        
        this.checkpointRadius = 15;
        this.checkpoints = [ [200,451], [170,451], [109,442], [105,416], [126,402], [167,400],
        [220,400], [244,384], [238,363], [202,347], [182,322], [165,238], [154,151], [116,123],
        [91,81],
        [340,398],
        [263,454] ];
        
        this.Reset();
    }
    
    IsDisqualified()
    {
        return this.disqualified;
    }
    
    Reset()
    {
        this.input_speed = 0;
        this.input_steer = 0;

        this.timeSinceBeginning = 0;
        this.disqualified = false;
        this.x = 263;
        this.y = 454;
        this.angle = Math.PI;
        this.speed = 0;
        this.nextThinkIn = this.thinkTimeS;
        this.reward = 0;
        this.checkpointId = 0;
    }

    Process(dt) {
        if ( this.disqualified)
            return;

        this.nextThinkIn -= dt;
        if (this.nextThinkIn <= 0)
        {
            this.nextThinkIn += this.thinkTimeS;
            this._Think();
        }

        this._Simulate(dt);

        if ( this._HasHitWall() ){
            this.disqualified = true;
        }

        this.timeSinceBeginning += dt;
    }

    Mutate(amount){
        this.brain.Mutate(amount);
    }

    DeepCopy(){
        return new Entity(this.pixelGetter, this.brain.DeepCopy());
    }

    _Simulate(dt){
        let speedModifier = dt * (2.0*(this.input_speed - 0.5));
        this.speed += speedModifier * (speedModifier < 0 ? this.carBrakeStrength : this.carAcceleration);

        let drag = this.speed * this.speed * this.carAirResistance;
        this.speed -= drag;

        if ( this.speed < 0 )
            this.speed = 0;

        this.angle += dt * this.carSteeringSpeed * (2.0*(this.input_steer - 0.5)) * this.speed;

        let deltaX = Math.cos(this.angle);
        let deltaY = -Math.sin(this.angle);

        this.x += deltaX * this.speed * dt;
        this.y += deltaY * this.speed * dt;

        this.reward += this.speed * this.speed * dt;

        let nextCheckpoint = this.checkpoints[this.checkpointId];
        if ( Math.abs( nextCheckpoint[0] - this.x ) < this.checkpointRadius && Math.abs( nextCheckpoint[1] - this.y ) < this.checkpointRadius ){
            this.checkpointId = (this.checkpointId+1) % this.checkpoints.length;
            this.reward += 1000;
        }
    }

    GenerationEnd(){
        //Any additional calculation for rewards can come here.
    }

    //Returns distance in scaled value. 0.0 means that the entity has hit a wall. 1.0 means that the entity cannot see a wall in this direction until the visibility distance
    _EyeDistanceScaled(relativeAngle){
        let testX = this.x;
        let testY = this.y;
        
        let deltaX = Math.cos(this.angle + relativeAngle);
        let deltaY = -Math.sin(this.angle + relativeAngle);

        let ret = 1.0;

        let step = 5;

        for(let i = 0; i <= this.maxSeeingDistancePx / step; i++){
            testX += deltaX * step;
            testY += deltaY * step;
            if (this._HasHitWallAtPoint(testX, testY)){
                ret = (i * step) / this.maxSeeingDistancePx;
                break;
            }
        }

        return ret;
    }

    _HasHitWall(){
        return this._HasHitWallAtPoint(this.x, this.y);
    }

    _HasHitWallAtPoint(x, y){
        let imgValue = this._GetImageValueAt(x, y);
        return imgValue > 0.2;
    }

    _Think(){
        let sensorData = [
            this._EyeDistanceScaled(0), //See forward,

            this._EyeDistanceScaled(-this.sensor_spreadView), //See left,
            this._EyeDistanceScaled(-this.sensor_spreadView * 2.0), //See left,

            this._EyeDistanceScaled(this.sensor_spreadView), //See right,
            this._EyeDistanceScaled(this.sensor_spreadView * 2.0) //See right,
            ];

        let input = [ this.speed, this.angle, this.input_speed, this.input_steer, ...sensorData ];

        [this.input_speed, this.input_steer] = this.brain.Calculate(input);

    }

    Draw(context){
        let size = 10;
        let spread = 2.5;

        let p1x = this.x + Math.cos(this.angle) * size;
        let p1y = this.y - Math.sin(this.angle) * size;
        
        let p2x = this.x + Math.cos(this.angle + spread) * size;
        let p2y = this.y - Math.sin(this.angle + spread) * size;

        let p3x = this.x + Math.cos(this.angle - spread) * size;
        let p3y = this.y - Math.sin(this.angle - spread) * size;
        
        let p4x = this.x + Math.cos(this.angle + spread) * size*0.4;
        let p4y = this.y - Math.sin(this.angle + spread) * size*0.4;

        let p5x = this.x + Math.cos(this.angle - spread) * size*0.4;
        let p5y = this.y - Math.sin(this.angle - spread) * size*0.4;

        context.fillStyle = this.IsDisqualified() ? "#771100" : "#FF3300";
        context.beginPath();
        context.moveTo(p1x, p1y);
        context.lineTo(p2x, p2y);
        context.lineTo(p3x, p3y);
        context.fill();
        
        context.fillStyle = this.IsDisqualified() ? "#775500" : "#FFCC00";
        context.beginPath();
        context.moveTo(p1x, p1y);
        context.lineTo(p4x, p4y);
        context.lineTo(p5x, p5y);
        context.fill();

    }

    _GetImageValueAt(x, y) {
        return this.pixelGetter(x, y) / 255.0;
    }
}