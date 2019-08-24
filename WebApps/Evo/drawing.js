

class EvoDrawing
{
    constructor(canvas, imgName, resultsName){
        this.canvas = canvas;
        this.context = canvas.getContext("2d");

        this.paused=false;
        this.resultsLabelName = resultsName;
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

        this.entityCountTarget = 60;
        this.entityTimeoutS = 500;
        this.frameTimeS = 0.05; 
        this.generationSurvivorPercentage = 0.2;
        this.learningRate = 0.01;
        this.outlierChance = 0.1;
        this.outlierDelta = 20;

        this.currentSessionTimer = 0;
        this.currentGeneration = 1;
        this.simulationSpeed = 1;
        
        while( this.entities.length < this.entityCountTarget ){
            this.entities.push(new Entity(this.GetPixelAtPoint.bind(this)));
        }
        
        this._MutateEntities();
        
        this.timerFnc = setInterval(()=>{this.Tick()}, (this.frameTimeS * 1000) / this.simulationSpeed);
    }

    _MutateEntities(){
        for(let entity of this.entities){
            let outlier = (Math.random() < this.outlierChance) ? this.outlierDelta : 0;
            entity.Mutate(this.learningRate);
        }
    }

    _Timeout() {
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

        this._MutateEntities();

        for(let entity of this.entities){
            entity.Reset();
        }
        
        this.currentSessionTimer = 0;
        this.currentGeneration++;
    }

    SetSimulationSpeed(speed){
        this.simulationSpeed = speed;
        clearInterval(this.timerFnc);
        this.timerFnc = setInterval(()=>{this.Tick()}, (this.frameTimeS * 1000) / this.simulationSpeed);
        this.paused = speed == 0;
    }

    Simulate(dt) {
        if ( this.paused )
            return;

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
        
        var resultsPanel = document.getElementById(this.resultsLabelName);

        resultsPanel.innerText = "Generation: " + this.currentGeneration + 
         "\nCars in race: " + this.entities.filter(x => !x.IsDisqualified()).length;
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

