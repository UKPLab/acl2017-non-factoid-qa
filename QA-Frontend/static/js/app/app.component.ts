import {Component, ViewChild, OnInit} from "@angular/core";
import {QAService} from "./qa.service";

export interface IAttentionOptions {
    showAttention: boolean,
    sensitivity: number,
    threshold: number,
    showTransparency: boolean,
}

interface IQuestion {
    tokens: string[]
}

interface IAnswer {
    tokens: string[],
    weights: number[],
    questionWeights: number[]
}

@Component({
    selector: 'my-app',
    templateUrl: 'static/templates/app.html'
})
export class AppComponent extends OnInit {
    viewStates: {
        questionAsked: boolean,
        showFakeInput: boolean,
        showComparison: boolean,
        loading: boolean,
        loadingComparison:boolean,
        attention: IAttentionOptions,
        visibleQuestionWeights: number[]
    } = {
        questionAsked: false,
        showFakeInput: false,
        showComparison: false,
        loading: false,
        loadingComparison: false,
        attention: {
            showAttention: true,
            sensitivity: 0.5,
            threshold: 0.5,
            showTransparency: true,
        },
        visibleQuestionWeights: []
    };

    data: {
        question: string,
        usedReranker: string,
        result: {
            question: IQuestion,
            answers: IAnswer[]
        },
        comparison: {
            a: {
                reranker: string,
                question: IQuestion,
                answer: IAnswer[]
            },
            b: {
                reranker: string,
                question: IQuestion,
                answer: IAnswer[]
            }
        }
        errorMessage: string,
        selectedReRanker: {label: string, url: string},
        reRankers: {label: string, url: string}[]
    } = {
        question: '',
        usedReranker: '',
        result: null,
        comparison: null,
        errorMessage: null,
        selectedReRanker: window['re_rankers'][0],
        reRankers: window['re_rankers']
    };

    @ViewChild('questionInput') questionInputVC;

    constructor(private qaService: QAService) {
        super()
    }

    ngOnInit() {
        this.questionInputVC.nativeElement.focus();
    }

    ask(question: string): void {
        window['$'](this.questionInputVC.nativeElement).popover('hide');

        this.viewStates.questionAsked = true;
        this.viewStates.loading = true;
        this.viewStates.visibleQuestionWeights = [];
        this.viewStates.showComparison = false;
        this.viewStates.showFakeInput = false;
        this.data.errorMessage = null;
        this.data.result = null;
        this.data.usedReranker = this.data.selectedReRanker.label;

        this.qaService.askQuestion(question, this.data.selectedReRanker.label)
            .subscribe(
                (result: any) => {
                    this.data.result = result;
                    this.viewStates.loading = false;
                    this.viewStates.showFakeInput = true;
                    if(result.answers.length > 0) {
                        this.viewStates.visibleQuestionWeights = result.answers[0].questionWeights;
                    }
                },
                error => {
                    this.data.errorMessage = <any>error;
                    this.viewStates.loading = false;
                }
            );
    }

    compare(answer:any, reranker:string) {
        this.viewStates.showComparison = true;
        this.viewStates.loadingComparison = true;
        let questionText = this.data.result.question.tokens.join(' ');
        let answerText = answer.tokens.join(' ');
        this.qaService.getWeights(questionText, answerText, reranker)
            .subscribe(
                (result: any) => {
                    this.viewStates.loadingComparison = false;
                    this.data.comparison = {
                        a: {
                            reranker: this.data.usedReranker,
                            question: this.data.result.question,
                            answer: answer
                        },
                        b: {
                            reranker: reranker,
                            question: result.question,
                            answer: result.candidate
                        }
                    };
                },
                error => {
                    this.data.errorMessage = <any>error;
                    this.viewStates.loadingComparison = false;
                    this.viewStates.showComparison = false;
                }
            );
    }

    focusInput() {
        this.viewStates.showFakeInput = false;
        setTimeout(() => {
            this.questionInputVC.nativeElement.focus();
            this.questionInputVC.nativeElement.setSelectionRange(0, this.questionInputVC.nativeElement.value.length);
        }, 50);
    }

}