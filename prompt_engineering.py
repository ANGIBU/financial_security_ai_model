# prompt_engineering.py
"""
고급 금융보안 특화 프롬프트 엔지니어링 시스템 - 최적화 버전
정확도 향상을 위한 정교한 프롬프트 설계
"""

import re
import random
from typing import Dict, List, Tuple, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class AdvancedPromptEngineer:
    """고급 프롬프트 엔지니어링 클래스 - 최적화 버전"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.expert_examples = self._build_expert_examples()
        self.optimized_templates = self._build_optimized_templates()
        
    def _build_expert_examples(self) -> Dict[str, List[Dict]]:
        """전문가 수준의 분석 예시"""
        examples = {
            "multiple_choice": [
                {
                    "question": """전자금융거래법상 전자금융거래의 정의로 가장 적절한 것은?
1 인터넷을 이용한 모든 거래
2 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래  
3 신용카드를 이용한 거래
4 ATM을 통한 현금 인출
5 모바일 앱을 통한 모든 서비스""",
                    "analysis": """전자금융거래법 제2조 제1호에 따르면 "전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래"가 정의입니다.
선택지 검토:
1번 - 인터넷만 한정(X)
2번 - 법적 정의와 일치(O)
3번 - 신용카드만 한정(X)
4번 - 현금인출만 한정(X)
5번 - 금융과 무관 포함(X)

정답: 2""",
                    "answer": "2"
                }
            ],
            "subjective": [
                {
                    "question": "금융기관의 개인정보보호 관리체계 구축 시 고려해야 할 핵심 요소들을 설명하시오.",
                    "answer": "개인정보보호 관리체계 구축 시 다음 요소들이 핵심입니다. 첫째, 개인정보보호 정책 수립과 시행 체계입니다. 둘째, 개인정보 처리 현황 파악과 정기 점검입니다. 셋째, 접근권한 관리 및 접근통제 시스템입니다. 넷째, 개인정보의 암호화입니다. 다섯째, 접근기록 보관 및 점검입니다."
                }
            ]
        }
        return examples
    
    def _build_optimized_templates(self) -> Dict[str, str]:
        """최적화된 분석 템플릿"""
        templates = {}
        
        # 간단한 객관식 템플릿
        templates["simple_mc"] = """문제: {question}

위 문제를 분석하여 정답 번호를 제시하세요.
핵심: 법령 정의나 명확한 사실에 근거하여 판단

정답:"""

        # 복잡한 객관식 템플릿
        templates["complex_mc"] = """당신은 금융보안 전문가입니다.

{knowledge_context}

문제: {question}

단계별 분석:
1. 핵심 개념 확인
2. 각 선택지 검토
3. 정답 결정

정답:"""

        # 부정형 문제 특화 템플릿
        templates["negative_mc"] = """⚠️ 부정형 문제 주의 ⚠️

문제: {question}

이 문제는 '해당하지 않는' 또는 '틀린' 것을 찾는 문제입니다.
각 선택지를 신중히 검토하여 명확히 틀린 것을 찾으세요.

선택지 분석 후 정답:"""

        # 주관식 간결 템플릿
        templates["subjective_concise"] = """질문: {question}

위 질문에 대해 핵심 내용을 중심으로 간결하고 정확하게 답변하세요.

답변:"""

        return templates
    
    def create_simple_mc_prompt(self, question: str) -> str:
        """간단한 객관식 프롬프트 (빠른 처리용)"""
        template = self.optimized_templates["simple_mc"]
        return template.format(question=question)
    
    def create_negative_mc_prompt(self, question: str) -> str:
        """부정형 문제 특화 프롬프트"""
        template = self.optimized_templates["negative_mc"]
        return template.format(question=question)
    
    def create_expert_prompt(self, question: str, question_type: str, 
                           strategy: str = "balanced") -> str:
        """전문가급 프롬프트 생성 - 최적화 버전"""
        
        # 간단한 문제는 빠른 템플릿 사용
        if strategy == "simple" and question_type == "multiple_choice":
            return self.create_simple_mc_prompt(question)
        
        # 지식 베이스 활용 (복잡한 문제만)
        knowledge_context = ""
        if strategy != "simple":
            analysis = self.knowledge_base.analyze_question(question)
            if analysis.get('relevant_laws') or analysis.get('related_concepts'):
                knowledge_context = self.knowledge_base.generate_analysis_context(question)
        
        if question_type == "multiple_choice":
            return self._create_mc_expert_prompt_optimized(question, knowledge_context)
        else:
            return self._create_subjective_expert_prompt_optimized(question)
    
    def _create_mc_expert_prompt_optimized(self, question: str, 
                                         knowledge_context: str) -> str:
        """최적화된 객관식 프롬프트"""
        
        # 부정형 체크
        if any(neg in question for neg in ['해당하지 않는', '적절하지 않은', '옳지 않은']):
            return self.create_negative_mc_prompt(question)
        
        # 복잡한 문제용 템플릿
        if knowledge_context:
            template = self.optimized_templates["complex_mc"]
            return template.format(
                knowledge_context=knowledge_context,
                question=question
            )
        else:
            # 간단한 템플릿
            return self.create_simple_mc_prompt(question)
    
    def _create_subjective_expert_prompt_optimized(self, question: str) -> str:
        """최적화된 주관식 프롬프트"""
        template = self.optimized_templates["subjective_concise"]
        return template.format(question=question)
    
    def create_few_shot_expert_prompt(self, question: str, question_type: str) -> str:
        """Few-shot 프롬프트 - 간소화 버전"""
        examples = self.expert_examples[question_type]
        
        # 가장 관련성 높은 예시 1개만 선택
        example = self._select_best_example(question, examples)
        
        prompt = f"""전문가 분석 예시:
문제: {example['question']}
{example.get('analysis', '')}
답: {example['answer']}

현재 문제:
{question}

위 예시처럼 분석하여 답하세요.
"""
        
        if question_type == "multiple_choice":
            prompt += "정답:"
        else:
            prompt += "답변:"
        
        return prompt
    
    def _select_best_example(self, question: str, examples: List[Dict]) -> Dict:
        """가장 적합한 예시 선택"""
        if not examples:
            return {"question": "", "answer": "", "analysis": ""}
        
        # 간단한 키워드 매칭
        question_lower = question.lower()
        best_score = 0
        best_example = examples[0]
        
        for example in examples:
            score = 0
            example_lower = example['question'].lower()
            
            # 공통 단어 수 계산
            question_words = set(question_lower.split())
            example_words = set(example_lower.split())
            common_words = question_words & example_words
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_example = example
        
        return best_example
    
    def create_chain_of_thought_prompt(self, question: str, question_type: str) -> str:
        """Chain-of-Thought 프롬프트 - 간소화"""
        
        if question_type == "multiple_choice":
            return f"""문제: {question}

사고 과정:
1. 문제 핵심 파악
2. 관련 지식 적용
3. 선택지 분석
4. 정답 결정

정답:"""
        else:
            return f"""질문: {question}

체계적으로 답변하세요:
1. 핵심 내용 정리
2. 구체적 설명
3. 결론

답변:"""
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 최적화 - 간소화"""
        if "solar" in model_name.lower():
            return f"### User:\n{prompt}\n\n### Assistant:\n"
        
        elif "llama" in model_name.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt