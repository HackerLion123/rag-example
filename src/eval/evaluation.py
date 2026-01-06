import json
from typing import List, Optional
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

from src.models.agent import create_agent, get_llm_client
from src.models.schemas import EvalCase, EvalResult
from src.models.prompt import create_grading_prompt, create_citation_eval_prompt
from src.models.reranker import create_reranker
import logging

logger = logging.getLogger(__name__)

EVAL_CASES = [
    EvalCase(id="case_001", question="what are few drills techiques?", category="general"),
    EvalCase(id="case_002", question="Tell me about llm agents", category="summary"),
    EvalCase(id="case_003", question="What are the key procedures mentioned for drilling?", category="summary"),
    EvalCase(id="case_004", question="when should we drill?", category="general"),
    EvalCase(id="case_005", question="What training materials are available?", category="training"),
    EvalCase(id="case_006", question="give a summary on drilling fluids", category="summary"),
    EvalCase(id="case_007", question="explain about bottom hole assembly?", category="summary"),
    EvalCase(id="case_008", question="explain about Cementing?", category="summary"),
    EvalCase(id="case_009", question="explain about drilling fluid additives", category="summary")
]


class LLMGrader:
    def __init__(self):
        self.llm = get_llm_client()
        self.grading_prompt = create_grading_prompt()
        self.citation_prompt = create_citation_eval_prompt()
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = create_reranker()
        return self._reranker

    def reranker_metrics(self, question: str, citations: List[dict]) -> dict:
        """Compute deterministic SentenceTransformer reranker metrics over the provided citations."""
        reranker = self._get_reranker()
        
        if not citations:
            return {
                "reranker_model": None,
                "reranker_top_score": 0.0,
                "reranker_mean_score": 0.0,
                "reranker_n": 0,
            }

        documents: List[Document] = []
        for idx, c in enumerate(citations):
            text = (c or {}).get("relevant_text") or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={"citation_index": idx}))

        if not documents:
            return {
                "reranker_model": reranker.model_name,
                "reranker_top_score": 0.0,
                "reranker_mean_score": 0.0,
                "reranker_n": 0,
            }

        
        ranked = reranker.rerank(question, documents)
        if not ranked:
            return {
                "reranker_model": reranker.model_name,
                "reranker_top_score": 0.0,
                "reranker_mean_score": 0.0,
                "reranker_n": 0,
            }

        scores = [r.score for r in ranked]
        top_score = max(scores) if scores else 0.0
        mean_score = (sum(scores) / len(scores)) if scores else 0.0
        return {
            "reranker_model": reranker.model_name,
            "reranker_top_score": round(float(top_score),2),
            "reranker_mean_score": round(float(mean_score),2),
            "reranker_n": len(scores),
        }

    def grade_response(
        self,
        question: str,
        expected: Optional[str],
        generated: str,
        citations: Optional[List[dict]] = None,
    ) -> dict:
        if not expected:
            expected = "A relevant and accurate answer based on the available documents."

        rr = self.reranker_metrics(question, citations or [])
        
        chain = self.grading_prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated,
                "reranker_top_score": rr["reranker_top_score"],
                "reranker_mean_score": rr["reranker_mean_score"],
            })
            result.update(rr)
            return result
        except Exception as e:
            return {
                "correctness": 0,
                "relevance": 0,
                "clarity": 0,
                "overall": 0,
                "passed": 0,
                "reason": "UNSUPPORTED",
                **self.reranker_metrics(question, citations or []),
                "error": f"Grading error: {str(e)}",
            }

    def grade_citations(self, answer: str, citations: List[dict]) -> dict:
        if not citations:
            return {"citation_relevance": 0, "citation_support": 0, "reason": "MISSING"}
        
        chain = self.citation_prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({
                "answer": answer,
                "citations": json.dumps(citations, indent=2)
            })
            return result
        except Exception:
            return {"citation_relevance": 0, "citation_support": 0, "reason": "UNSUPPORTED"}


class EvaluationSuite:   
    def __init__(self, cases: Optional[List[EvalCase]] = None):
        self.cases = cases or EVAL_CASES
        self.grader = LLMGrader()
        self.agent = None

    def _get_agent(self):
        if self.agent is None:
            self.agent = create_agent()
        return self.agent

    def run_case(self, case: EvalCase) -> EvalResult:
        agent = self._get_agent()
        response = agent.query(case.question)
        
        scores = self.grader.grade_response(
            case.question,
            case.expected_answer,
            response["answer"],
            citations=response.get("citations", []),
        )
        
        try:
            citation_scores = self.grader.grade_citations(
                response["answer"],
                response["citations"]
            )
            scores.update(citation_scores)
        except:
            logger.warning(f"Failed to grade citations for case {case.id}")
        passed_val = scores.get("passed", scores.get("overall", 0))
        passed = bool(passed_val == 1 or passed_val is True)
        
        return EvalResult(
            case_id=case.id,
            question=case.question,
            generated_answer=response["answer"],
            expected_answer=case.expected_answer,
            citations=response["citations"],
            confidence=response["confidence"],
            scores=scores,
            passed=passed,
            failure_reason=scores.get("reason") if not passed else None
        )

    def run_all(self) -> List[EvalResult]:
        results = []
        for case in self.cases:
            result = self.run_case(case)
            results.append(result)
        return results

    def generate_report(self, results: List[EvalResult]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        avg_scores = {}
        score_keys = ["correctness", "completeness", "relevance", "clarity", "overall", 
                      "citation_relevance", "citation_support"]
        
        for key in score_keys:
            values = [r.scores.get(key, 0) for r in results if key in r.scores]
            avg_scores[key] = sum(values) / len(values) if values else 0
        
        category_results = {}
        for result in results:
            case = next((c for c in self.cases if c.id == result.case_id), None)
            if case:
                cat = case.category
                if cat not in category_results:
                    category_results[cat] = {"passed": 0, "failed": 0}
                if result.passed:
                    category_results[cat]["passed"] += 1
                else:
                    category_results[cat]["failed"] += 1
        
        failures = [
            {
                "case_id": r.case_id,
                "question": r.question,
                "reason": r.failure_reason,
                "scores": r.scores
            }
            for r in results if not r.passed
        ]
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_cases": total,
                "version": "1.0.0"
            },
            "summary": {
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0
            },
            "average_scores": avg_scores,
            "category_breakdown": category_results,
            "failure_analysis": failures,
            "detailed_results": [r.model_dump() for r in results]
        }


def run_evaluation(output_path: Optional[str] = None) -> dict:
    suite = EvaluationSuite()
    results = suite.run_all()
    report = suite.generate_report(results)
    print(report)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    
    return report


if __name__ == "__main__":
    report = run_evaluation("evaluation_report.json")
    print(f"Evaluation complete: {report['summary']['passed']}/{report['metadata']['total_cases']} passed")
