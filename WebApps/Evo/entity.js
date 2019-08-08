class Entity {
    constructor() {
        this.brain = new NeuralNetwork(GetInitialNetwork());
        this.disqualified = false;
        this.imgWidth = 512;
        this.imgHeight = 512;

        this.x = 263;
        this.y = 454;
        this.angle = Math.PI;
        this.speed = 0;
        this.thinkTime = 0.1;
        this.nextThinkIn = this.thinkTime;

        this.input_speed = 0;
        this.input_steer = 0;

        this.carSteeringSpeed = 0.1;
        this.carAcceleration = 10.0;
        this.carBrakeStrength = 1.0;

        this.sensor_spreadView = 0.01;
        this.sensor_dist_near= 40;
        this.sensor_dist_far= 70;

        this.reward = 0;

        console.log(this._GetAngleFromCenterPoint());

    }

    _GetAngleFromCenterPoint(){
        return Math.atan2(this.x - this.imgWidth*0.5, this.y - this.imgHeight*0.5);
    }

    Process(dt) {
        if ( this.disqualified)
            return;

        this.nextThinkIn -= dt;
        if (this.nextThinkIn < 0)
        {
            this.nextThinkIn += this.thinkTime;
            this._Think();
        }

        this._Simulate(dt);

        if ( this._HasHitWall() ){
            this.reward = 0;
            this.disqualified = true;
        }
    }

    _Simulate(dt){
        let speedModifier = dt * (2.0*(this.input_speed - 0.5));
        this.speed += speedModifier * (speedModifier < 0 ? this.carBrakeStrength : this.carAcceleration);
        if ( this.speed < 0 )
            this.speed = 0;

        this.angle += dt * this.carSteeringSpeed * (2.0*(this.input_steer - 0.5)) * this.speed;

        let deltaX = Math.cos(this.angle);
        let deltaY = -Math.sin(this.angle);

        this.x += deltaX * this.speed * dt;
        this.x += deltaY * this.speed * dt;

        this.reward += this.speed * dt;


    }

    _HasHitWall(){
        let imgValue = this._GetImageValueAt(this.x, this.y);
        return imgValue > 0.2;
    }

    _Think(){
        let deltaX = Math.cos(this.angle);
        let deltaY = -Math.sin(this.angle);
        
        let deltaX_left = Math.cos(this.angle -  this.spreadView);
        let deltaY_left = -Math.sin(this.angle - this.spreadView);
        
        let deltaX_right = Math.cos(this.angle +  this.spreadView);
        let deltaY_right = -Math.sin(this.angle + this.spreadView);

        let sensors = [
            this._GetImageValueAt(this.x + deltaX * this.sensor_dist_near, this.y+deltaY * this.sensor_dist_near), //See forward 
            this._GetImageValueAt(this.x + deltaX * this.sensor_dist_far, this.y+deltaY * this.sensor_dist_far), //See forward further
            this._GetImageValueAt(this.x + deltaX_left * this.sensor_dist_near, this.y+deltaY_left * this.sensor_dist_near), //See left 
            this._GetImageValueAt(this.x + deltaX_right * this.sensor_dist_near, this.y+deltaY_right * this.sensor_dist_near), //See right
            ];

        let input = [ this.speed, this.angle, this.input_speed, this.input_steer, ...sensors ];

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

        context.fillStyle = "#FF3300";
        context.beginPath();
        context.moveTo(p1x, p1y);
        context.lineTo(p2x, p2y);
        context.lineTo(p3x, p3y);
        context.fill();

        context.fillStyle = "#FFCC00";
        context.beginPath();
        context.moveTo(p1x, p1y);
        context.lineTo(p4x, p4y);
        context.lineTo(p5x, p5y);
        context.fill();

    }

    _GetImageValueAt(x, y){
        return 0.0;
    }
}