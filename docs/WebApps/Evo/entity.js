var drawSensorsGlobal = false;

class Entity {
    constructor(pixelGetter, neuralNetwork = null) {
        if (neuralNetwork === null)
            this.brain = new NeuralNetwork(GetInitialNetwork());
        else
            this.brain = neuralNetwork;
        this.pixelGetter = pixelGetter;
        this.imgWidth = 512;
        this.imgHeight = 512;
        
        this.maxSeeingDistancePx = 150;

        this.thinkTimeS = 0.05;

        this.carSteeringSpeed = 0.25;
        this.carAcceleration = 5.0;
        this.carBrakeStrength = 20.0;
        this.carAirResistance = 0.0001;
        
        this.sensor_spreadView = 0.45;

        this.drawSensors = false;
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

        this.lowSpeedCounter = 0;
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

        if ( this._HasHitWall() || this.lowSpeedCounter > 2.0 ){
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

        this.reward += this.speed * this.speed * dt * 0.01;

        if (this.speed < 0.3){
            this.lowSpeedCounter += dt;
        } else {
            this.lowSpeedCounter = 0.0;
        }

        let nextCheckpoint = this.checkpoints[this.checkpointId];
        if ( Math.abs( nextCheckpoint[0] - this.x ) < this.checkpointRadius && Math.abs( nextCheckpoint[1] - this.y ) < this.checkpointRadius ){
            this.checkpointId = (this.checkpointId+1) % this.checkpoints.length;
            this.reward += 10000;
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

    _GetSensorData(){
        return [
            this._EyeDistanceScaled(0), //See forward,

            this._EyeDistanceScaled(-this.sensor_spreadView), //See left,
            this._EyeDistanceScaled(-this.sensor_spreadView * 2.0), //See left,

            this._EyeDistanceScaled(this.sensor_spreadView), //See right,
            this._EyeDistanceScaled(this.sensor_spreadView * 2.0) //See right,
            ];
    }

    _Think(){
        let sensorData = this._GetSensorData(); 

        let input = [ this.speed, this.angle, this.input_speed, this.input_steer, ...sensorData ];

        [this.input_speed, this.input_steer] = this.brain.Calculate(input);

    }

    Draw(context){
        let size = 10;
        let spread = 2.5;
        
        if ((this.drawSensors || drawSensorsGlobal) && !this.IsDisqualified()){
            let sensorData = this._GetSensorData(); 

            let s1x = this.x + Math.cos(this.angle) * sensorData[0] * this.maxSeeingDistancePx;
            let s1y = this.y - Math.sin(this.angle) * sensorData[0] * this.maxSeeingDistancePx;

            let s2x = this.x + Math.cos(this.angle-this.sensor_spreadView) * sensorData[1] * this.maxSeeingDistancePx;
            let s2y = this.y - Math.sin(this.angle-this.sensor_spreadView) * sensorData[1] * this.maxSeeingDistancePx;

            let s3x = this.x + Math.cos(this.angle-this.sensor_spreadView*2) * sensorData[2] * this.maxSeeingDistancePx;
            let s3y = this.y - Math.sin(this.angle-this.sensor_spreadView*2) * sensorData[2] * this.maxSeeingDistancePx;

            let s4x = this.x + Math.cos(this.angle+this.sensor_spreadView) * sensorData[3] * this.maxSeeingDistancePx;
            let s4y = this.y - Math.sin(this.angle+this.sensor_spreadView) * sensorData[3] * this.maxSeeingDistancePx;

            let s5x = this.x + Math.cos(this.angle+this.sensor_spreadView*2) * sensorData[4] * this.maxSeeingDistancePx;
            let s5y = this.y - Math.sin(this.angle+this.sensor_spreadView*2) * sensorData[4] * this.maxSeeingDistancePx;

            context.strokeStyle = "#2288ff";
            context.lineWidth = 1;
            context.beginPath();
            context.moveTo(this.x, this.y);
            context.lineTo(s1x, s1y);
            context.moveTo(this.x, this.y);
            context.lineTo(s2x, s2y);
            context.moveTo(this.x, this.y);
            context.lineTo(s3x, s3y);
            context.moveTo(this.x, this.y);
            context.lineTo(s4x, s4y);
            context.moveTo(this.x, this.y);
            context.lineTo(s5x, s5y);
            context.stroke();
        }

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