import {Directive, ElementRef, Input, Output, EventEmitter} from "@angular/core";

@Directive({selector: '[bsSlider]'})
export class SliderDirective {
    slider: any;

    @Input() sliderValue: number;
    @Output() sliderValueChange = new EventEmitter();

    constructor(el: ElementRef) {
        this.slider = window['$'](el.nativeElement).slider({
            value: this.sliderValue
        });
        this.slider.on('change', (e) => {
            this.sliderValue = this.slider.slider('getValue');
            this.sliderValueChange.emit(this.sliderValue);
        });
    }
}

@Directive({selector: '[bsTooltip]'})
export class TooltipDirective {
    constructor(el: ElementRef) {
        window['$'](el.nativeElement).tooltip();
    }
}
