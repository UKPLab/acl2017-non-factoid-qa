import {Injectable} from "@angular/core";
import {Http, Response, URLSearchParams} from "@angular/http";
import "rxjs/add/operator/catch";
import "rxjs/add/operator/map";
import {Observable} from "rxjs/Observable";
import {Answer} from "./qa.models";

@Injectable()
export class QAService {
    private askUrl = '/get-answers';
    private getWeightsUrl = '/weights';

    constructor(private http: Http) {

    }

    askQuestion(question: string, reranker: string): Observable<Answer[]> {
        let params = new URLSearchParams();
        params.set('re_ranker', reranker);
        params.set('q', question);

        return this.http.get(this.askUrl, {search: params})
            .map(this.extractData)
            .catch(this.handleError);
    }

    getWeights(questionText: string, candidateText: string, reRanker: string) {
        let params = new URLSearchParams();
        params.set('re_ranker', reRanker);

        return this.http.post(this.getWeightsUrl, {question: questionText, candidate: candidateText}, {search: params})
            .map(this.extractData)
            .catch(this.handleError);
    }

    private extractData(res: Response) {
        let body = res.json();
        return body || null;
    }

    private handleError(error: Response, caught: Observable<any>): any {
        throw new Error(error.text());
    }
}