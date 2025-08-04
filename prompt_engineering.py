# prompt_engineering.py
"""
고급 금융보안 특화 프롬프트 엔지니어링 시스템
실제 분석과 추론을 강제하는 전문가급 프롬프트
"""

import re
import random
from typing import Dict, List, Tuple, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class AdvancedPromptEngineer:
    """고급 프롬프트 엔지니어링 클래스"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.expert_examples = self._build_expert_examples()
        self.analysis_templates = self._build_analysis_templates()
        
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
                    "analysis": """
**단계별 분석:**

1. **법적 근거 확인:**
   - 전자금융거래법 제2조 제1호: "전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래"

2. **핵심 요소 식별:**
   - 전자적 장치 이용 필수
   - 금융상품/서비스 관련
   - 제공과 이용의 쌍방향성

3. **선택지 분석:**
   - 1번: 인터넷만으로 한정 (X)
   - 2번: 법정 정의와 정확히 일치 (O)
   - 3번: 신용카드만으로 제한 (X)  
   - 4번: 현금인출은 금융서비스 제공이 아님 (X)
   - 5번: 모든 서비스는 범위 과대 (X)

4. **법리적 판단:**
   전자금융거래법 제2조 제1호의 명문 규정에 따라 2번이 정답""",
                    "answer": "2"
                },
                {
                    "question": """개인정보보호법상 개인정보 유출 발견 시 신고 시한으로 옳은 것은?
1 즉시
2 3일 이내  
3 7일 이내
4 30일 이내
5 신고 의무 없음""",
                    "analysis": """
**단계별 분석:**

1. **법적 근거 확인:**
   - 개인정보보호법 제34조 제1항: "개인정보 유출 등을 안 때에는 지체 없이 신고"

2. **법해석 원칙:**
   - "지체 없이" = 즉시, 지연 없이
   - 구체적 일수 규정 없음

3. **선택지 검토:**
   - 1번: "즉시"는 "지체 없이"와 동일한 의미 (O)
   - 2-4번: 구체적 기간은 법문에 없음 (X)
   - 5번: 명문으로 신고 의무 규정 (X)

4. **법리적 결론:**
   "지체 없이" = "즉시"가 법적으로 정확한 해석""",
                    "answer": "1"
                }
            ],
            "subjective": [
                {
                    "question": "금융기관의 개인정보보호 관리체계 구축 시 고려해야 할 핵심 요소들을 설명하시오.",
                    "analysis": """
**체계적 분석:**

1. **법적 기반:**
   - 개인정보보호법 제29조(안전성 확보조치)
   - 개인정보보호법 시행령 제30조

2. **핵심 구성요소:**
   - 정책 및 절차 수립
   - 조직 및 인력 관리
   - 기술적 보호조치
   - 관리적 보호조치

3. **구체적 요소들:**
   - 개인정보보호 정책 수립 및 시행
   - 개인정보 처리 현황 파악 및 점검
   - 접근권한 관리 및 접근통제
   - 개인정보의 암호화
   - 접근기록의 보관 및 점검""",
                    "answer": "개인정보보호 관리체계 구축 시에는 다음 핵심 요소들을 고려해야 합니다. 첫째, 개인정보보호 정책 수립과 시행 체계를 마련해야 합니다. 둘째, 개인정보 처리 현황을 정확히 파악하고 정기적으로 점검하는 체계가 필요합니다. 셋째, 접근권한 관리 및 접근통제 시스템을 통해 권한이 있는 자만 개인정보에 접근할 수 있도록 해야 합니다. 넷째, 개인정보의 암호화를 통해 기술적 보호조치를 강화해야 합니다. 다섯째, 접근기록의 보관 및 점검을 통해 개인정보 처리 활동을 모니터링해야 합니다."
                }
            ]
        }
        return examples
    
    def _build_analysis_templates(self) -> Dict[str, str]:
        """분석 템플릿 구축"""
        templates = {}
        
        templates["deep_analysis_mc"] = """당신은 금융보안원 FSKU 평가 전문가입니다.

{knowledge_context}

**문제:**
{question}

**전문가 분석 절차:**

1. **법적/이론적 근거 확인:**
   - 관련 법령 및 조항 식별
   - 핵심 개념 정의 확인

2. **문제 핵심 파악:**
   - 문제에서 묻는 정확한 내용
   - 부정형/긍정형 문제 구분

3. **선택지별 정밀 분석:**
   {choices_analysis}

4. **전문가 판단:**
   - 법리적 근거와 실무적 관점 종합
   - 가장 정확하고 적절한 답 결정

**분석 결과:**
위 절차에 따라 단계별로 분석하고 최종 답번호만 제시하세요.

정답:"""

        templates["expert_subjective"] = """당신은 금융보안 분야 최고 전문가입니다.

{knowledge_context}

**질문:** {question}

**전문가 답변 구성:**

1. **핵심 개념 정의**
2. **법적 근거 및 규정**
3. **실무적 적용 방안**
4. **중요 고려사항**

각 단계별로 정확하고 전문적인 내용을 포함하여 답변하세요.

**전문가 답변:**"""

        templates["law_focused"] = """당신은 금융법 전문가입니다.

{knowledge_context}

**법령 기반 분석:**

문제: {question}

**분석 단계:**
1. 관련 법령 및 조항 특정
2. 법문 해석 및 적용
3. 선택지별 법적 타당성 검토
4. 최종 법리적 결론

정답:"""

        return templates
    
    def create_expert_prompt(self, question: str, question_type: str, 
                           strategy: str = "deep_analysis") -> str:
        """전문가급 프롬프트 생성"""
        
        # 지식 베이스에서 관련 정보 추출
        knowledge_context = self.knowledge_base.generate_analysis_context(question)
        analysis = self.knowledge_base.analyze_question(question)
        
        if question_type == "multiple_choice":
            return self._create_mc_expert_prompt(question, knowledge_context, analysis, strategy)
        else:
            return self._create_subjective_expert_prompt(question, knowledge_context, analysis)
    
    def _create_mc_expert_prompt(self, question: str, knowledge_context: str, 
                                analysis: Dict, strategy: str) -> str:
        """객관식 전문가 프롬프트"""
        
        # 선택지 추출 및 분석
        question_text, choices = self._extract_question_and_choices(question)
        choices_analysis = self._generate_choices_analysis_guide(choices, analysis)
        
        # 전략별 템플릿 선택
        if strategy == "law_focused" and analysis.get('relevant_laws'):
            template = self.analysis_templates["law_focused"]
            return template.format(
                knowledge_context=knowledge_context,
                question=question
            )
        else:
            template = self.analysis_templates["deep_analysis_mc"]
            return template.format(
                knowledge_context=knowledge_context,
                question=question,
                choices_analysis=choices_analysis
            )
    
    def _create_subjective_expert_prompt(self, question: str, knowledge_context: str, 
                                       analysis: Dict) -> str:
        """주관식 전문가 프롬프트"""
        template = self.analysis_templates["expert_subjective"]
        return template.format(
            knowledge_context=knowledge_context,
            question=question
        )
    
    def _generate_choices_analysis_guide(self, choices: List[str], analysis: Dict) -> str:
        """선택지 분석 가이드 생성"""
        guide_lines = []
        
        for i, choice in enumerate(choices, 1):
            guide_lines.append(f"   선택지 {i}: {choice}")
            guide_lines.append(f"   → 분석: [법적 근거, 개념 정확성, 실무 적합성 검토]")
        
        # 부정형 문제 특별 안내
        if analysis.get('negative_question'):
            guide_lines.append("\n   ⚠️ 부정형 문제: '해당하지 않는' 또는 '틀린' 것을 찾으세요!")
        
        return "\n".join(guide_lines)
    
    def create_few_shot_expert_prompt(self, question: str, question_type: str) -> str:
        """Few-shot 전문가 프롬프트"""
        examples = self.expert_examples[question_type]
        
        # 관련성 높은 예시 선택
        relevant_examples = self._select_relevant_examples(question, examples)
        
        prompt = "당신은 금융보안원 FSKU 평가 전문가입니다. 다음 전문가 분석 예시를 참고하세요.\n\n"
        
        for i, example in enumerate(relevant_examples, 1):
            prompt += f"**전문가 분석 예시 {i}:**\n"
            prompt += f"문제: {example['question']}\n"
            prompt += f"{example['analysis']}\n"
            prompt += f"정답: {example['answer']}\n\n"
        
        # 현재 문제
        knowledge_context = self.knowledge_base.generate_analysis_context(question)
        prompt += f"**현재 분석 문제:**\n\n{knowledge_context}\n\n"
        
        if question_type == "multiple_choice":
            prompt += "위 전문가 분석 예시와 같은 방식으로 단계별 분석 후 정답 번호만 제시하세요.\n\n"
            prompt += f"문제: {question}\n\n분석 및 정답:"
        else:
            prompt += "위 전문가 답변 수준으로 전문적이고 정확한 답변을 제공하세요.\n\n"
            prompt += f"질문: {question}\n\n전문가 답변:"
        
        return prompt
    
    def _select_relevant_examples(self, question: str, examples: List[Dict]) -> List[Dict]:
        """관련성 높은 예시 선택"""
        question_analysis = self.knowledge_base.analyze_question(question)
        scored_examples = []
        
        for example in examples:
            score = 0
            example_analysis = self.knowledge_base.analyze_question(example['question'])
            
            # 도메인 일치도
            common_domains = set(question_analysis['domain']) & set(example_analysis['domain'])
            score += len(common_domains) * 2
            
            # 문제 유형 일치도
            if question_analysis['question_type'] == example_analysis['question_type']:
                score += 3
            
            # 부정형 문제 일치도
            if question_analysis['negative_question'] == example_analysis['negative_question']:
                score += 1
            
            scored_examples.append((score, example))
        
        # 점수 순으로 정렬하여 상위 2개 선택
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:2]]
    
    def create_chain_of_thought_prompt(self, question: str, question_type: str) -> str:
        """Chain-of-Thought 프롬프트"""
        knowledge_context = self.knowledge_base.generate_analysis_context(question)
        analysis = self.knowledge_base.analyze_question(question)
        
        if question_type == "multiple_choice":
            question_text, choices = self._extract_question_and_choices(question)
            
            prompt = f"""당신은 금융보안 전문가입니다.

{knowledge_context}

문제: {question_text}

선택지:
{chr(10).join(f"{i+1}. {choice}" for i, choice in enumerate(choices))}

**전문가 사고과정:**

단계 1: 문제 핵심 파악
- 이 문제에서 묻는 것은 무엇인가?
- 어떤 법령이나 개념과 관련이 있는가?

단계 2: 관련 지식 활용  
- 해당 분야의 정확한 정의나 규정은?
- 실무에서 적용되는 원칙은?

단계 3: 선택지 검증
- 각 선택지가 법적, 기술적으로 정확한가?
- 문제에서 요구하는 조건을 만족하는가?

단계 4: 최종 판단
- 가장 정확하고 적절한 답은?

위 사고과정을 거쳐 분석하고 정답 번호를 제시하세요."""

        else:
            prompt = f"""당신은 금융보안 전문가입니다.

{knowledge_context}

질문: {question}

**전문가 사고과정:**

단계 1: 질문 분석
- 핵심 키워드와 요구사항 파악
- 관련 법령 및 규정 확인

단계 2: 지식 체계화
- 관련 개념들의 정의와 관계
- 실무적 적용 사례

단계 3: 답변 구성
- 논리적 순서로 내용 정리
- 정확하고 완전한 설명

위 과정을 통해 전문적이고 정확한 답변을 제공하세요."""

        return prompt
    
    def _extract_question_and_choices(self, full_text: str) -> Tuple[str, List[str]]:
        """질문과 선택지 분리"""
        lines = full_text.strip().split("\n")
        question_lines = []
        choices = []
        
        choice_pattern = r"^\s*[1-5]\s+"
        
        for line in lines:
            line = line.strip()
            if line and re.match(choice_pattern, line):
                choices.append(line)
            elif line:
                question_lines.append(line)
        
        question = " ".join(question_lines)
        return question, choices
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 최적화"""
        if "solar" in model_name.lower():
            system_prompt = "당신은 금융보안원 FSKU 평가 전문가입니다. 정확한 분석과 논리적 추론을 통해 답변합니다."
            return f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        
        elif "llama" in model_name.lower():
            system_prompt = "You are an expert in financial security evaluation with deep knowledge of Korean financial regulations."
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt