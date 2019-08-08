

class EvoDrawing
{
    constructor(canvas){
        this.canvas = canvas;
        this.context = canvas.getContext("2d");

        this.entities = [];

        this.entityCountTarget = 1;
        this.entitySurvivals = 1;
        this.entityTimeoutS = 20;
        this.frameTimeS = 0.05; 

        this.currentSessionTimer = 0;
        
        while( this.entities.length < this.entityCountTarget ){
            this.entities.push(new Entity());
        }
        
        setInterval(()=>{this.Tick()}, this.frameTimeS * 1000);
    }

    _GenerationEnd(){
        for(let entity of this.entities){
            entity.GenerationEnd();
        }
    }

    _Timeout(){
        this._GenerationEnd();
    }

    Simulate(dt) {
        this.currentSessionTimer += dt;
        if ( this.currentSessionTimer > this.entityTimeoutS){
            _Timeout();
        }

        for(let entity of this.entities){
            entity.Process(dt);
        }
    }

    Draw() {
        var img = document.getElementById("background");
        this.context.drawImage(img, 0, 0);
        
        for(let entity of this.entities){
            entity.Draw(this.context);
        }
    }

    Tick(){
        this.Simulate(this.frameTimeS);
        this.Draw();
    }
}


var drawing = new EvoDrawing(document.getElementById("drawCanvas"));