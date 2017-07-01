import {Component, Input} from "@angular/core";
import {IAttentionOptions} from "./app.component";

@Component({
    selector: 'weighted-text',
    template: `<span *ngFor="let token of tokens; let i = index"
                      [ngStyle]="{'background-color': backgroundColor(weights[i]), 'opacity': opacity(weights[i])}"
                      bsTooltip
                      [attr.data-original-title]="'Importance: ' + weights[i]">
                    {{ token }}
                </span>`
})
export class WeightedText {
    @Input() tokens: string[];
    @Input() weights: number[];
    @Input() attentionOptions: IAttentionOptions;

    private stats = {min: 0, max: 0, avg: 0, std: 0};

    ngOnInit() {

    }

    ngOnChanges(changes: any) {
        let jStat = window['jStat'];
        let weightsFiltered = this.weights.filter((w: number) => w > 0);
        var avgWeight = jStat.median(weightsFiltered);
        var maxWeight = jStat.max(weightsFiltered);
        var minWeight = jStat.min(weightsFiltered);
        var stdWeight = jStat.stdev(weightsFiltered);
        this.stats = {avg: avgWeight, max: maxWeight, std: stdWeight, min: minWeight};
    }

    backgroundColor(weight: number): string {
        if (!this.attentionOptions.showAttention) {
            return 'hsla(0, 100%, 68.8%, 0)';
        } else {
            let weightMaxStd = Math.min(this.stats.max - this.stats.avg, this.stats.std);
            let alpha = Math.max(0, this.attentionOptions.sensitivity * (weight - this.stats.avg) / weightMaxStd);
            if (weight < Math.min(this.stats.max, this.stats.avg + weightMaxStd * this.attentionOptions.threshold)) {
                return 'hsla(0, 100%, 68.8%, 0)';
            } else {
                return `hsla(0, 100%, 68.8%, ${alpha})`;
            }
        }
    }

    opacity(weight: number): string {
        if (!this.attentionOptions.showTransparency || !this.attentionOptions.showAttention) {
            return '1.0';
        } else {
            var alpha = 1 - Math.min(0.5, (weight - this.stats.avg) / (this.stats.min - this.stats.avg) * this.attentionOptions.sensitivity);
            if (weight > this.stats.avg - Math.min((this.stats.avg - this.stats.min) / 2, this.stats.std) * this.attentionOptions.threshold) {
                return '1.0';
            } else {
                return `${alpha}`;
            }
        }
    }
}