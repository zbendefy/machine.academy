

class EvoDrawing
{
    constructor(canvas, imgName){
        this.canvas = canvas;
        this.context = canvas.getContext("2d");

        this.trackImageName = imgName;
        let img = document.getElementById(imgName);
        this.offscreenCanvas = document.createElement('canvas');
        this.imgWidth = img.width;
        this.offscreenCanvas.width = img.width;
        this.offscreenCanvas.height = img.height;
        this.offscreenCanvas.getContext('2d').drawImage(img, 0, 0, img.width, img.height);
        this.offscreenCanvasContext = this.offscreenCanvas.getContext('2d');
        let offscreenCanvasImageRGBA = this.offscreenCanvasContext.getImageData(0, 0, img.width, img.height);
        this.offscreenCanvasImage = [];
        for(let iy = 0; iy < img.height; iy++){
            for(let ix = 0; ix < img.width; ix++){
                this.offscreenCanvasImage[iy * img.width + ix] = ( offscreenCanvasImageRGBA.data[(iy * img.width + ix)*4] );
            }   
        }

        this.entities = [];

        this.entityCountTarget = 50;
        this.entityTimeoutS = 20;
        this.frameTimeS = 0.05; 
        this.generationSurvivorPercentage = 0.2;
        this.learningRate = 0.01;

        this.currentSessionTimer = 0;
        this.currentGeneration = 1;
        this.simulationSpeed = 4;
        
        while( this.entities.length < this.entityCountTarget ){
            this.entities.push(new Entity(this.GetPixelAtPoint.bind(this)));
        }
        
        for(let entity of this.entities){
            entity.Mutate(this.learningRate);
        }
        
        setInterval(()=>{this.Tick()}, (this.frameTimeS * 1000) / this.simulationSpeed);
    }

    _Timeout() {
        console.log("Generation " + this.currentGeneration + " finished. Rewards:");

        for(let entity of this.entities){
            entity.GenerationEnd();
        }

        this.entities.sort( function(a,b){ return b.reward-a.reward; } ); //Sort by descending order

        this.entities = this.entities.slice(0, this.entities.length * this.generationSurvivorPercentage);
        
        let survivorCount = this.entities.length;
        let currentEntityIdx = 0;
        while(this.entities.length < this.entityCountTarget){
            this.entities.push(this.entities[currentEntityIdx].DeepCopy());
            currentEntityIdx = (currentEntityIdx+1)%survivorCount;
        }

        for(let entity of this.entities){
            entity.Mutate(this.learningRate);
        }

        for(let entity of this.entities){
            entity.Reset();
        }
        
        this.currentSessionTimer = 0;
        this.currentGeneration++;
    }

    Simulate(dt) {
        let allEntitiesDisqualified = this.entities.every( (entity)=>{return entity.IsDisqualified()} );

        this.currentSessionTimer += dt;
        if ( this.currentSessionTimer > this.entityTimeoutS || allEntitiesDisqualified){
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
        return this.offscreenCanvasImage[(y|0)*this.imgWidth+(x|0)];
    }
}


var drawing = new EvoDrawing(document.getElementById("drawCanvas"), "background");