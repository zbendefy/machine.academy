

class EvoDrawing
{
    constructor(canvas, imgName){
        this.canvas = canvas;
        this.context = canvas.getContext("2d");

        this.trackImageName = imgName;
        let img = document.getElementById(imgName);
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCanvas.width = img.width;
        this.offscreenCanvas.height = img.height;
        this.offscreenCanvas.getContext('2d').drawImage(img, 0, 0, img.width, img.height);

        this.entities = [];

        this.entityCountTarget = 1;
        this.entitySurvivals = 1;
        this.entityTimeoutS = 20;
        this.frameTimeS = 0.05; 

        this.currentSessionTimer = 0;
        this.currentGeneration = 1;
        
        while( this.entities.length < this.entityCountTarget ){
            this.entities.push(new Entity(this.GetPixelAtPoint.bind(this)));
        }
        
        setInterval(()=>{this.Tick()}, this.frameTimeS * 1000);
    }

    _Timeout() {
        for(let entity of this.entities){
            entity.GenerationEnd();
        }

        this.entities.sort( function(a,b){ return b.reward-a.reward; } ); //Sort by descending order

        this.entities.slice(this.entities.length/2);
        
        //TODO: copy and mutate elements
        
        this.currentSessionTimer = 0;
        this.currentGeneration++;
    }

    Simulate(dt) {
        this.currentSessionTimer += dt;
        if ( this.currentSessionTimer > this.entityTimeoutS){
            this._Timeout();
        }

        for(let entity of this.entities){
            entity.Process(dt);
        }
    }

    Draw() {
        var img = document.getElementById(this.trackImageName);
        this.context.drawImage(img, 0, 0);
        
        for(let entity of this.entities){
            entity.Draw(this.context);
        }
    }

    Tick(){
        this.Simulate(this.frameTimeS);
        this.Draw();
    }

    GetPixelAtPoint(x,y)
    {
        let pixelValue = this.offscreenCanvas.getContext("2d").getImageData(x,y,1,1);
        return pixelValue.data[0];
    }
}


var drawing = new EvoDrawing(document.getElementById("drawCanvas"), "background");