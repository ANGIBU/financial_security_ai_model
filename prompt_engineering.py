# prompt_engineering.py

"""
프롬프트 엔지니어링
- 객관식/주관식 프롬프트 생성
- Chain-of-Thought 추론 프롬프트
- 단계별 추론 유도 프롬프트
- 논리적 검증 프롬프트
- 다중 관점 분석 프롬프트
- 도메인별 템플릿 관리
- 한국어 강화 프롬프트
"""

import gc
import hashlib
import re
from typing import Dict, List, Optional, Tuple

from knowledge_base import FinancialSecurityKnowledgeBase

# 상수 정의
DEFAULT_CACHE_SIZE = 200
CACHE_CLEANUP_INTERVAL = 50
PROMPT_MAX_LENGTH = 2000
TEMPLATE_CACHE_SIZE = 20

class PromptEngineer:
    
    def __init__(self):
        """프롬프트 엔지니어링 시스템 초기화"""
        try:
            self.knowledge_base = FinancialSecurityKnowledgeBase()
            self.templates = self._build_optimized_templates()
            self.cot_templates = self._build_cot_templates()
            self.reasoning_templates = self._build_reasoning_templates()
            
            # 캐시 시스템
            self.prompt_cache = {}
            self.template_cache = {}
            self.max_cache_size = DEFAULT_CACHE_SIZE
            self.cache_cleanup_counter = 0
            
            # 통계 추적
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "template_usage": {},
                "domain_distribution": {},
                "prompt_generations": 0,
                "cot_usage": 0,
                "reasoning_usage": 0,
                "avg_prompt_length": 0.0
            }
            
        except Exception as e:
            raise RuntimeError(f"프롬프트 엔지니어 초기화 실패: {e}")
    
    def _build_optimized_templates(self) -> Dict[str, str]:
        """최적화된 기본 템플릿 구축"""
        templates = {}
        
        # 객관식 기본 템플릿
        templates["mc_basic"] = """{question}

위 문제의 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 객관식 부정형 템플릿
        templates["mc_negative"] = """{question}

이 문제는 틀린 것, 해당하지 않는 것, 적절하지 않은 것을 찾는 문제입니다.
정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 금융투자업 특화 템플릿
        templates["mc_financial"] = """{question}

금융투자업 분류:
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 금융투자업 아님: 소비자금융업, 보험중개업

정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 주관식 기본 템플릿
        templates["subj_basic"] = """{question}

위 질문에 대해 한국어로 답변하세요.
법령과 규정에 따른 구체적인 설명을 포함하세요."""

        # 사이버보안 특화 템플릿
        templates["subj_trojan"] = """{question}

트로이 목마의 특징과 탐지 방법에 대해 한국어로 설명하세요."""

        # 개인정보보호 특화 템플릿
        templates["subj_personal_info"] = """{question}

개인정보보호법에 따른 조치사항을 한국어로 설명하세요."""

        # 전자금융 특화 템플릿
        templates["subj_electronic"] = """{question}

전자금융거래법에 따른 안전성 확보 방안을 한국어로 설명하세요."""

        # 정보보안 특화 템플릿
        templates["subj_info_security"] = """{question}

정보보안 관리체계에 따른 체계적인 관리 방안을 한국어로 설명하세요."""

        # 위험관리 특화 템플릿
        templates["subj_risk_management"] = """{question}

위험관리 체계 구축 방안을 한국어로 설명하세요."""

        return templates
    
    def _build_cot_templates(self) -> Dict[str, str]:
        """Chain-of-Thought 템플릿 구축"""
        templates = {}
        
        # 객관식 CoT 템플릿
        templates["mc_cot"] = """{question}

이 문제를 단계별로 분석해보겠습니다:

1단계: 문제 이해
- 질문의 핵심을 파악합니다
- 부정형 질문인지 확인합니다

2단계: 관련 개념 검토
- 관련된 법령과 개념을 검토합니다
- 각 선택지의 의미를 분석합니다

3단계: 논리적 추론
- 개념과 법령을 바탕으로 추론합니다
- 각 선택지를 검증합니다

4단계: 답 선택
정답: """

        # 주관식 CoT 템플릿  
        templates["subj_cot"] = """{question}

이 질문에 대해 단계별로 분석하여 답변하겠습니다:

1단계: 문제 분석
- 질문의 핵심 요구사항을 파악합니다

2단계: 관련 법령 및 개념 검토
- 적용 가능한 법령과 규정을 확인합니다
- 핵심 개념과 정의를 검토합니다

3단계: 체계적 답변 구성
- 논리적 순서로 답변을 구성합니다

답변: """

        # 부정형 CoT 템플릿
        templates["mc_cot_negative"] = """{question}

이 부정형 문제를 단계별로 분석해보겠습니다:

1단계: 부정형 질문 확인
- "해당하지 않는 것", "틀린 것", "적절하지 않은 것" 등을 찾는 문제입니다

2단계: 각 선택지 검증
- 각 선택지가 올바른지 틀린지 판단합니다
- 관련 법령과 개념을 바탕으로 검증합니다

3단계: 틀린 것 식별
- 논리적 배제법을 사용합니다
- 가장 부적절한 것을 선택합니다

정답: """

        # 개념 정의 CoT 템플릿
        templates["definition_cot"] = """{question}

이 개념 정의 문제를 단계별로 분석해보겠습니다:

1단계: 개념 파악
- 정의하고자 하는 개념을 명확히 합니다

2단계: 법적 근거 확인
- 관련 법령에서의 정의를 확인합니다
- 핵심 구성 요소를 파악합니다

3단계: 논리적 결론
- 법적 정의와 개념적 특성을 종합합니다

답변: """

        return templates
    
    def _build_reasoning_templates(self) -> Dict[str, str]:
        """논리적 추론 템플릿 구축"""
        templates = {}
        
        # 다중 관점 분석 템플릿
        templates["multi_perspective"] = """{question}

이 문제를 다양한 관점에서 분석해보겠습니다:

관점 1: 법률적 관점
- 관련 법령과 규정의 요구사항

관점 2: 기술적 관점  
- 기술적 구현과 보안 고려사항

관점 3: 관리적 관점
- 조직과 운영 관리 측면

종합 분석:
각 관점을 종합하여 최적의 답을 도출합니다.

답변: """

        # 논리적 검증 템플릿
        templates["logical_verification"] = """{question}

논리적 검증을 통해 답변을 도출해보겠습니다:

가정 설정:
- 주어진 조건과 전제를 명확히 합니다

논리적 추론:
- 전제로부터 논리적 결론을 도출합니다
- 각 단계의 타당성을 검증합니다

결론 검증:
- 도출된 결론이 논리적으로 타당한지 확인합니다
- 반례나 모순이 없는지 검토합니다

최종 답변: """

        # 단계별 추론 템플릿
        templates["step_by_step"] = """{question}

단계별 추론 과정:

전제 확인:
- 문제에서 주어진 정보와 조건을 정리합니다

추론 과정:
- 전제로부터 중간 결론들을 단계적으로 도출합니다
- 각 단계가 논리적으로 연결되는지 확인합니다

최종 결론:
- 모든 추론 단계를 종합하여 최종 답을 도출합니다

답변: """

        # 대조 분석 템플릿
        templates["comparative_analysis"] = """{question}

대조 분석을 통한 문제 해결:

유사 개념 비교:
- 관련된 유사 개념들과의 차이점을 분석합니다

예외 사항 검토:
- 일반 원칙에서 벗어나는 예외 상황을 확인합니다

적용 범위 분석:
- 각 개념이나 규정의 적용 범위를 명확히 합니다

결론:
대조 분석을 바탕으로 정확한 답을 선택합니다.

답변: """

        return templates
    
    def create_cot_prompt(self, question: str, question_type: str, 
                         analysis: Optional[Dict] = None) -> str:
        """Chain-of-Thought 프롬프트 생성"""
        try:
            if not question or not question.strip():
                raise ValueError("질문이 비어있습니다")
            
            if len(question) > PROMPT_MAX_LENGTH:
                question = question[:PROMPT_MAX_LENGTH-3] + "..."
            
            # 캐시 키 생성
            cache_content = f"cot_{question[:200]}{question_type}"
            cache_key = hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:16]
            
            if cache_key in self.prompt_cache:
                self.stats["cache_hits"] += 1
                return self.prompt_cache[cache_key]
            
            self.stats["cache_misses"] += 1
            self.stats["cot_usage"] += 1
            
            question_lower = question.lower()
            
            # CoT 템플릿 선택
            if question_type == "multiple_choice":
                if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
                    template_key = "mc_cot_negative"
                elif "정의" in question_lower or "의미" in question_lower:
                    template_key = "definition_cot"
                else:
                    template_key = "mc_cot"
            else:
                template_key = "subj_cot"
            
            # 템플릿 적용
            if template_key not in self.cot_templates:
                template_key = "mc_cot" if question_type == "multiple_choice" else "subj_cot"
            
            prompt = self.cot_templates[template_key].format(question=question.strip())
            
            # 통계 업데이트
            self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
            self.stats["prompt_generations"] += 1
            
            # 캐시 관리 및 저장
            self._manage_cache()
            self.prompt_cache[cache_key] = prompt
            
            return prompt
            
        except Exception as e:
            return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_reasoning_prompt(self, question: str, reasoning_type: str, 
                              analysis: Optional[Dict] = None) -> str:
        """논리적 추론 프롬프트 생성"""
        try:
            if not question or not question.strip():
                raise ValueError("질문이 비어있습니다")
            
            if len(question) > PROMPT_MAX_LENGTH:
                question = question[:PROMPT_MAX_LENGTH-3] + "..."
            
            # 캐시 키 생성
            cache_content = f"reasoning_{reasoning_type}_{question[:200]}"
            cache_key = hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:16]
            
            if cache_key in self.prompt_cache:
                self.stats["cache_hits"] += 1
                return self.prompt_cache[cache_key]
            
            self.stats["cache_misses"] += 1
            self.stats["reasoning_usage"] += 1
            
            # 추론 템플릿 선택
            template_key = reasoning_type
            if template_key not in self.reasoning_templates:
                template_key = "step_by_step"  # 기본값
            
            prompt = self.reasoning_templates[template_key].format(question=question.strip())
            
            # 도메인별 추가 정보 삽입
            if analysis and "domain" in analysis:
                domain_context = self._get_domain_context(analysis["domain"])
                if domain_context:
                    prompt = prompt.replace("답변: ", f"{domain_context}\n\n답변: ")
            
            # 통계 업데이트
            self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
            self.stats["prompt_generations"] += 1
            
            # 캐시 관리 및 저장
            self._manage_cache()
            self.prompt_cache[cache_key] = prompt
            
            return prompt
            
        except Exception as e:
            return self.create_korean_reinforced_prompt(question, "subjective")
    
    def create_multi_perspective_prompt(self, question: str, perspectives: List[str]) -> str:
        """다중 관점 분석 프롬프트 생성"""
        try:
            base_template = self.reasoning_templates["multi_perspective"]
            
            # 관점별 상세 내용 추가
            perspective_details = {
                "법률적": "관련 법령의 규정과 의무사항을 검토합니다",
                "기술적": "기술적 구현 방법과 보안 고려사항을 분석합니다", 
                "관리적": "조직 운영과 관리 체계 측면을 고려합니다",
                "실무적": "실제 현장에서의 적용과 운영 방안을 검토합니다",
                "정책적": "정책 수립과 제도 개선 관점에서 분석합니다"
            }
            
            # 선택된 관점들로 템플릿 커스터마이징
            if len(perspectives) >= 2:
                perspective_sections = []
                for i, perspective in enumerate(perspectives[:3], 1):
                    detail = perspective_details.get(perspective, f"{perspective} 관점에서의 분석")
                    perspective_sections.append(f"관점 {i}: {perspective} 관점\n- {detail}")
                
                custom_template = question + "\n\n이 문제를 다양한 관점에서 분석해보겠습니다:\n\n"
                custom_template += "\n\n".join(perspective_sections)
                custom_template += "\n\n종합 분석:\n각 관점을 종합하여 최적의 답을 도출합니다.\n\n답변: "
                
                return custom_template
            else:
                return base_template.format(question=question)
                
        except Exception:
            return self.create_korean_reinforced_prompt(question, "subjective")
    
    def create_verification_prompt(self, question: str, initial_answer: str) -> str:
        """답변 검증 프롬프트 생성"""
        try:
            verification_template = """질문: {question}

초기 답변: {initial_answer}

위 답변을 검증해보겠습니다:

1단계: 논리적 일관성 검사
- 답변의 논리적 흐름이 일관된가?
- 전제와 결론이 올바르게 연결되었는가?

2단계: 사실 정확성 검증
- 언급된 법령과 규정이 정확한가?
- 기술적 내용이 올바른가?

3단계: 완성도 평가
- 질문에 충분히 답변했는가?
- 빠진 중요한 내용은 없는가?

검증 결과:
초기 답변이 적절한지, 수정이 필요한지 판단하고 개선된 답변을 제시합니다.

최종 답변: """

            return verification_template.format(
                question=question.strip(),
                initial_answer=initial_answer.strip()
            )
            
        except Exception:
            return f"다음 답변을 검증해주세요: {initial_answer}"
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        """한국어 강화 프롬프트 생성 (기존 메서드 유지)"""
        if not question or not question.strip():
            raise ValueError("질문이 비어있습니다")
        
        if len(question) > PROMPT_MAX_LENGTH:
            question = question[:PROMPT_MAX_LENGTH-3] + "..."
        
        # 캐시 키 생성
        cache_content = f"{question[:200]}{question_type}{len(question)}"
        cache_key = hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:16]
        
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        try:
            question_lower = question.lower()
            
            # 템플릿 선택 로직
            if question_type == "multiple_choice":
                template_key = self._select_mc_template(question_lower)
            else:
                template_key = self._select_subj_template(question_lower)
            
            # 템플릿 적용
            if template_key not in self.templates:
                template_key = "mc_basic" if question_type == "multiple_choice" else "subj_basic"
            
            prompt = self.templates[template_key].format(question=question.strip())
            
            # 통계 업데이트
            self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
            self.stats["prompt_generations"] += 1
            
            # 평균 프롬프트 길이 업데이트
            total_length = self.stats["avg_prompt_length"] * (self.stats["prompt_generations"] - 1)
            total_length += len(prompt)
            self.stats["avg_prompt_length"] = total_length / self.stats["prompt_generations"]
            
            # 캐시 관리
            self._manage_cache()
            self.prompt_cache[cache_key] = prompt
            
            return prompt
            
        except Exception as e:
            # 오류 발생 시 기본 템플릿 사용
            fallback_key = "mc_basic" if question_type == "multiple_choice" else "subj_basic"
            return self.templates[fallback_key].format(question=question.strip())
    
    def _select_mc_template(self, question_lower: str) -> str:
        """객관식 템플릿 선택"""
        # 금융투자업 관련 특수 처리
        if ("금융투자업" in question_lower and 
            ("소비자금융업" in question_lower or "보험중개업" in question_lower)):
            return "mc_financial"
        
        # 부정형 질문 처리
        negative_patterns = ["해당하지", "적절하지", "옳지", "틀린", "잘못된", "부적절한"]
        if any(pattern in question_lower for pattern in negative_patterns):
            return "mc_negative"
        
        return "mc_basic"
    
    def _select_subj_template(self, question_lower: str) -> str:
        """주관식 템플릿 선택"""
        # 도메인별 특화 템플릿 선택
        domain_keywords = {
            "subj_trojan": ["트로이", "악성코드", "멀웨어", "바이러스"],
            "subj_personal_info": ["개인정보", "정보주체", "개인정보보호법"],
            "subj_electronic": ["전자금융", "전자적", "접근매체", "전자금융거래법"],
            "subj_info_security": ["정보보안", "보안관리", "ISMS", "보안정책"],
            "subj_risk_management": ["위험관리", "위험평가", "위험분석", "위험통제"]
        }
        
        for template_key, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return template_key
        
        return "subj_basic"
    
    def _get_domain_context(self, domains: List[str]) -> str:
        """도메인 컨텍스트 정보 생성"""
        if not domains:
            return ""
        
        primary_domain = domains[0]
        
        context_info = {
            "개인정보보호": "개인정보보호법의 원칙과 정보주체의 권리를 고려하여",
            "전자금융": "전자금융거래법의 안전성 확보 의무와 이용자 보호를 중심으로",
            "정보보안": "정보보안 관리체계(ISMS)의 요구사항에 따라",
            "사이버보안": "사이버 위협에 대한 체계적 대응과 예방 관점에서",
            "위험관리": "위험 식별, 평가, 대응의 체계적 접근을 통해",
            "금융투자업": "자본시장법의 투자자 보호와 시장 건전성 확보 원칙에 따라"
        }
        
        return context_info.get(primary_domain, "관련 법령과 원칙에 따라")
    
    def create_progressive_hint_prompt(self, question: str, difficulty_level: str) -> str:
        """단계별 힌트 프롬프트 생성"""
        try:
            hint_levels = {
                "basic": "기본 개념과 정의를 중심으로",
                "intermediate": "관련 법령과 절차를 포함하여", 
                "advanced": "복합적 관점과 심화 분석을 통해"
            }
            
            hint_prefix = hint_levels.get(difficulty_level, "체계적 접근을 통해")
            
            progressive_template = f"""{question}

{hint_prefix} 이 문제를 해결해보겠습니다:

힌트 1: 핵심 키워드 파악
- 문제에서 중요한 개념이나 용어를 찾아보세요

힌트 2: 관련 법령 연결  
- 해당 영역의 주요 법령을 떠올려보세요

힌트 3: 논리적 추론
- 개념과 법령을 바탕으로 논리적으로 접근해보세요

답변: """

            return progressive_template
            
        except Exception:
            return self.create_korean_reinforced_prompt(question, "subjective")
    
    def create_self_correction_prompt(self, question: str, reasoning_chain: str) -> str:
        """자기 수정 프롬프트 생성"""
        try:
            correction_template = """질문: {question}

추론 과정: {reasoning_chain}

위 추론 과정을 자체 검토해보겠습니다:

검토 항목:
1. 논리적 비약이나 오류는 없는가?
2. 관련 법령이나 개념이 정확히 적용되었는가?
3. 결론이 전제와 일관되게 도출되었는가?

자기 수정:
발견된 문제점을 수정하고 개선된 답변을 제시합니다.

수정된 답변: """

            return correction_template.format(
                question=question.strip(),
                reasoning_chain=reasoning_chain.strip()
            )
            
        except Exception:
            return f"다음 추론을 검토하고 수정해주세요: {reasoning_chain}"
    
    def _manage_cache(self) -> None:
        """캐시 관리"""
        self.cache_cleanup_counter += 1
        
        if self.cache_cleanup_counter % CACHE_CLEANUP_INTERVAL == 0:
            # 프롬프트 캐시 정리
            if len(self.prompt_cache) >= self.max_cache_size:
                keys_to_remove = list(self.prompt_cache.keys())[:self.max_cache_size // 3]
                for key in keys_to_remove:
                    del self.prompt_cache[key]
            
            # 템플릿 캐시 정리
            if len(self.template_cache) >= TEMPLATE_CACHE_SIZE:
                keys_to_remove = list(self.template_cache.keys())[:TEMPLATE_CACHE_SIZE // 2]
                for key in keys_to_remove:
                    del self.template_cache[key]
            
            # 메모리 정리
            gc.collect()
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Optional[Dict] = None, 
                     structure: Optional[Dict] = None) -> str:
        """범용 프롬프트 생성 (호환성 유지)"""
        return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_few_shot_prompt(self, question: str, question_type: str, 
                              analysis: Optional[Dict] = None, 
                              num_examples: int = 1) -> str:
        """Few-shot 프롬프트 생성"""
        try:
            prompt_parts = []
            
            if question_type == "multiple_choice":
                # 객관식 예시
                prompt_parts.append("예시: 개인정보의 정의로 가장 적절한 것은?")
                prompt_parts.append("정답: 2")
                prompt_parts.append("")
            else:
                # 주관식 예시
                prompt_parts.append("예시: 트로이 목마의 특징을 설명하세요.")
                prompt_parts.append("답변: 트로이 목마는 정상 프로그램으로 위장한 악성코드입니다.")
                prompt_parts.append("")
            
            prompt_parts.append(f"문제: {question}")
            
            if question_type == "multiple_choice":
                prompt_parts.append("정답:")
            else:
                prompt_parts.append("답변:")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            return self.create_korean_reinforced_prompt(question, question_type)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 프롬프트 최적화"""
        if not prompt or not model_name:
            return prompt
        
        try:
            model_name_lower = model_name.lower()
            
            # SOLAR 모델 최적화
            if "solar" in model_name_lower:
                return f"### User:\n{prompt}\n\n### Assistant:\n"
            
            # LLaMA 모델 최적화
            elif "llama" in model_name_lower:
                return f"[INST] {prompt} [/INST]"
            
            # Mistral 모델 최적화
            elif "mistral" in model_name_lower:
                return f"<s>[INST] {prompt} [/INST]"
            
            # 기본 포맷
            else:
                return prompt
                
        except Exception:
            return prompt
    
    def get_template_suggestions(self, question: str) -> List[Tuple[str, float]]:
        """질문에 적합한 템플릿 제안"""
        suggestions = []
        question_lower = question.lower()
        
        try:
            # CoT 템플릿 적합도
            cot_score = 0.0
            if any(pattern in question_lower for pattern in ["분석", "단계", "과정"]):
                cot_score = 0.8
            elif len(question_lower) > 100:  # 복잡한 질문
                cot_score = 0.6
            
            if cot_score > 0.5:
                suggestions.append(("cot", cot_score))
            
            # 추론 템플릿 적합도
            reasoning_score = 0.0
            if any(pattern in question_lower for pattern in ["검토", "검증", "판단"]):
                reasoning_score = 0.7
            elif any(pattern in question_lower for pattern in ["비교", "대조"]):
                reasoning_score = 0.6
                suggestions.append(("comparative_analysis", reasoning_score))
            
            if reasoning_score > 0.5:
                suggestions.append(("logical_verification", reasoning_score))
            
            # 기본 템플릿 적합도
            basic_score = 0.5  # 기본값
            suggestions.append(("basic", basic_score))
            
            # 점수 순으로 정렬
            suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
            
        except Exception:
            suggestions = [("basic", 0.5)]
        
        return suggestions[:3]
    
    def get_stats_report(self) -> Dict:
        """통계 보고서 생성"""
        total_prompts = self.stats["prompt_generations"]
        cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        
        report = {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(cache_requests, 1),
            "cache_size": len(self.prompt_cache),
            "template_usage": dict(self.stats["template_usage"]),
            "domain_distribution": dict(self.stats["domain_distribution"]),
            "avg_prompt_length": self.stats["avg_prompt_length"],
            "cot_usage": self.stats["cot_usage"],
            "reasoning_usage": self.stats["reasoning_usage"],
            "advanced_features": {
                "cot_usage_rate": self.stats["cot_usage"] / max(total_prompts, 1),
                "reasoning_usage_rate": self.stats["reasoning_usage"] / max(total_prompts, 1)
            },
            "cache_efficiency": {
                "prompt_cache_size": len(self.prompt_cache),
                "template_cache_size": len(self.template_cache),
                "cleanup_count": self.cache_cleanup_counter
            }
        }
        
        # 가장 많이 사용된 템플릿
        if self.stats["template_usage"]:
            most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
            report["most_used_template"] = {"name": most_used[0], "count": most_used[1]}
        
        return report
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """프롬프트 유효성 검증"""
        issues = []
        
        if not prompt:
            issues.append("프롬프트가 비어있음")
            return False, issues
        
        if len(prompt) < 10:
            issues.append("프롬프트가 너무 짧음")
        
        if len(prompt) > PROMPT_MAX_LENGTH:
            issues.append(f"프롬프트가 너무 김 (최대 {PROMPT_MAX_LENGTH}자)")
        
        # 한국어 비율 확인
        korean_chars = len([c for c in prompt if '가' <= c <= '힣'])
        total_chars = len([c for c in prompt if c.isalnum()])
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            if korean_ratio < 0.3:
                issues.append("한국어 비율이 낮음")
        
        # 문제 문자 확인
        if re.search(r'[\u4e00-\u9fff]', prompt):
            issues.append("중국어 문자 포함")
        
        # CoT 구조 검증
        if "단계" in prompt and ":" not in prompt:
            issues.append("CoT 구조가 불완전함")
        
        return len(issues) == 0, issues
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 통계 요약
            total_usage = sum(self.stats["template_usage"].values())
            if total_usage > 0:
                most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
                cache_efficiency = self.stats["cache_hits"] / max(
                    self.stats["cache_hits"] + self.stats["cache_misses"], 1
                )
            
            # 캐시 정리
            self.prompt_cache.clear()
            self.template_cache.clear()
            
            # 통계 정리
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "template_usage": {},
                "domain_distribution": {},
                "prompt_generations": 0,
                "cot_usage": 0,
                "reasoning_usage": 0,
                "avg_prompt_length": 0.0
            }
            
            # 메모리 정리
            gc.collect()
            
        except Exception as e:
            pass