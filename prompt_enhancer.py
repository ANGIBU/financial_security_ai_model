# prompt_enhancer.py

import re
from typing import Dict, List
from config import OPTIMIZATION_CONFIG, POSITIONAL_ANALYSIS


class PromptEnhancer:
    """프롬프트 최적화"""

    def __init__(self):
        self.optimization_config = OPTIMIZATION_CONFIG
        self.positional_config = POSITIONAL_ANALYSIS
        self._initialize_templates()

    def _initialize_templates(self):
        """템플릿 초기화"""
        
        # 기본 프롬프트 템플릿
        self.base_templates = {
            "multiple_choice": {
                "system_prompt": "당신은 금융보안 전문가입니다. 객관식 문제를 정확하게 분석하여 정답을 선택하세요.",
                "instruction": "다음 객관식 문제를 분석하고 정답 번호만 제시하세요.",
                "analysis_steps": [
                    "1. 질문의 핵심 키워드 파악",
                    "2. 부정형/긍정형 문제 유형 확인", 
                    "3. 관련 법령 조항 검토",
                    "4. 각 선택지의 타당성 평가",
                    "5. 논리적 추론을 통한 정답 도출"
                ]
            },
            "subjective": {
                "system_prompt": "당신은 금융보안 전문가입니다. 주관식 문제에 대해 정확하고 전문적인 한국어 답변을 작성하세요.",
                "instruction": "다음 주관식 문제에 대해 관련 법령을 근거로 전문적인 한국어 답변을 작성하세요.",
                "requirements": [
                    "모든 답변은 한국어로만 작성",
                    "관련 법령의 구체적 조항 명시",
                    "실무적이고 구체적인 내용 포함",
                    "정확한 전문용어 사용",
                    "논리적 구조로 답변 구성"
                ]
            }
        }

        # 도메인별 특화 프롬프트
        self.domain_specific_templates = {
            "사이버보안": {
                "context": "사이버보안 위협 및 대응 기술에 대한 전문 지식",
                "keywords": ["악성코드", "트로이목마", "RAT", "SBOM", "딥페이크", "탐지지표"],
                "focus": "기술적 특징, 탐지 방법, 대응 방안"
            },
            "전자금융": {
                "context": "전자금융거래법 및 관련 규정",
                "keywords": ["전자금융거래법", "분쟁조정", "한국은행", "접근매체", "정보기술부문"],
                "focus": "법적 근거, 기관 역할, 비율 기준"
            },
            "개인정보보호": {
                "context": "개인정보보호법 및 관련 규정",
                "keywords": ["개인정보보호법", "정보주체", "법정대리인", "만 14세", "동의"],
                "focus": "처리 원칙, 주체 권리, 보호 조치"
            },
            "정보보안": {
                "context": "정보보안관리체계 및 기술적 보안",
                "keywords": ["ISMS", "3대요소", "재해복구", "접근통제", "암호화"],
                "focus": "관리체계, 기술적 보안, 복구 계획"
            },
            "금융투자": {
                "context": "자본시장법 및 금융투자업 규제",
                "keywords": ["금융투자업", "자본시장법", "적합성원칙", "투자자보호"],
                "focus": "업무 구분, 규제 체계, 투자자 보호"
            },
            "위험관리": {
                "context": "위험관리 체계 및 절차",
                "keywords": ["위험관리", "위험평가", "위험대응", "내부통제"],
                "focus": "위험 식별, 평가 방법, 대응 전략"
            },
            "정보통신": {
                "context": "정보통신기반 보호법 및 기술",
                "keywords": ["정보통신기반", "SPF", "프로토콜", "국내대리인"],
                "focus": "법적 의무, 기술 표준, 보고 절차"
            },
            "기타": {
                "context": "금융보안 관련 법령 및 규정",
                "keywords": ["법", "조", "규정", "기준", "관리", "운영"],
                "focus": "법적 근거, 관리 체계, 운영 원칙"
            }
        }

        # 위치별 프롬프트 조정
        self.position_adjustments = {
            "early": {
                "emphasis": "기본 개념과 원칙을 정확히 적용",
                "complexity": "standard",
                "detail_level": "medium"
            },
            "middle": {
                "emphasis": "실무 적용과 구체적 절차에 주목",
                "complexity": "moderate",
                "detail_level": "medium"
            },
            "late": {
                "emphasis": "법령 조항과 세부 기준을 정밀하게 검토",
                "complexity": "high",
                "detail_level": "high"
            }
        }

    def build_enhanced_prompt(self, question: str, question_type: str, domain: str = "일반",
                            context_info: str = "", institution_info: str = "",
                            force_diversity: bool = False, question_number: int = None) -> str:
        """향상된 프롬프트 생성"""
        
        try:
            # 위치 단계 확인
            position_stage = self._get_position_stage(question_number)
            
            # 기본 템플릿 선택
            base_template = self.base_templates.get(question_type, self.base_templates["subjective"])
            
            # 도메인 특화 정보
            domain_template = self.domain_specific_templates.get(domain, self.domain_specific_templates["기타"])
            
            # 위치별 조정
            position_adjustment = self.position_adjustments.get(position_stage, self.position_adjustments["middle"])
            
            # 프롬프트 구성
            prompt = self._construct_optimized_prompt(
                base_template, domain_template, position_adjustment,
                question, question_type, domain, context_info,
                institution_info, force_diversity, question_number
            )
            
            return prompt
            
        except Exception as e:
            print(f"프롬프트 생성 오류: {e}")
            return self._get_fallback_prompt(question, question_type, domain)

    def _get_position_stage(self, question_number: int) -> str:
        """위치 단계 확인"""
        if question_number is None:
            return "middle"
            
        if question_number <= 100:
            return "early"
        elif question_number <= 300:
            return "middle"
        else:
            return "late"

    def _construct_optimized_prompt(self, base_template: Dict, domain_template: Dict,
                                  position_adjustment: Dict, question: str, question_type: str,
                                  domain: str, context_info: str, institution_info: str,
                                  force_diversity: bool, question_number: int) -> str:
        """최적화된 프롬프트 구성"""
        
        prompt_parts = []
        
        # 시스템 프롬프트
        prompt_parts.append(base_template["system_prompt"])
        
        # 위치별 강조사항
        if position_adjustment["emphasis"]:
            prompt_parts.append(f"\n**위치별 주의사항**: {position_adjustment['emphasis']}")
        
        # 도메인 특화 컨텍스트
        if domain != "일반":
            prompt_parts.append(f"\n**전문 분야**: {domain_template['context']}")
            
            if domain_template["focus"]:
                prompt_parts.append(f"**중점 확인 사항**: {domain_template['focus']}")
        
        # 추가 컨텍스트 정보
        if context_info:
            prompt_parts.append(f"\n**참고 정보**:\n{context_info}")
        
        if institution_info:
            prompt_parts.append(f"\n**기관 정보**:\n{institution_info}")
        
        # 질문 유형별 지침
        if question_type == "multiple_choice":
            prompt_parts.append(self._get_mc_specific_instructions(question, question_number))
        else:
            prompt_parts.append(self._get_subjective_specific_instructions(question, question_number, force_diversity))
        
        # 질문 제시
        prompt_parts.append(f"\n**문제**: {question}")
        
        # 최종 지침
        if question_type == "multiple_choice":
            prompt_parts.append("\n위 지침에 따라 체계적으로 분석한 후 정답 번호만 제시하세요.\n\n정답:")
        else:
            prompt_parts.append("\n위 지침에 따라 관련 법령과 규정을 정확히 인용하면서 전문적인 한국어 답변을 작성하세요.\n\n한국어 답변:")
        
        return " ".join(prompt_parts)

    def _get_mc_specific_instructions(self, question: str, question_number: int = None) -> str:
        """객관식 특화 지침"""
        
        instructions = []
        
        # 기본 분석 방법
        instructions.append("\n**분석 방법**:")
        instructions.append("1. 질문 유형 판별: 부정형('해당하지 않는', '적절하지 않은', '옳지 않은') vs 긍정형('가장 적절한', '올바른')")
        instructions.append("2. 핵심 키워드 식별 및 관련 법령 확인")
        instructions.append("3. 각 선택지의 타당성을 법령과 규정에 따라 검토")
        instructions.append("4. 논리적 추론을 통한 정답 도출")
        
        # 위치별 특화 지침
        if question_number is not None:
            if question_number > 300:
                instructions.append("5. 후반부 문제 특성상 법령 조항과 세부 기준을 더욱 정밀하게 검토")
                instructions.append("6. 예외 사항과 특별 규정도 함께 고려")
        
        # 패턴별 힌트
        if "해당하지 않는" in question.lower() or "적절하지 않은" in question.lower():
            instructions.append("\n**부정형 문제**: 조건에 맞지 않는 선택지를 찾으세요.")
        elif "가장 적절한" in question.lower() or "가장 옳은" in question.lower():
            instructions.append("\n**긍정형 문제**: 조건에 가장 부합하는 선택지를 선택하세요.")
        
        return "\n".join(instructions)

    def _get_subjective_specific_instructions(self, question: str, question_number: int = None, force_diversity: bool = False) -> str:
        """주관식 특화 지침"""
        
        instructions = []
        
        # 기본 요구사항
        instructions.append("\n**필수 지침**:")
        instructions.append("1. 모든 답변은 한국어로만 작성 (영어 사용 절대 금지)")
        instructions.append("2. 관련 법령의 구체적 조항과 근거 명시")
        instructions.append("3. 실무적이고 구체적인 내용 포함")
        instructions.append("4. 정확한 전문용어 사용")
        
        # 위치별 특화 지침
        if question_number is not None:
            if question_number > 300:
                instructions.append("5. 후반부 문제 특성상 더욱 정밀하고 세부적인 답변 작성")
                instructions.append("6. 법령 조항 번호와 구체적 기준을 명확히 제시")
            elif question_number <= 100:
                instructions.append("5. 기본 개념과 원칙을 명확하게 설명")
            else:
                instructions.append("5. 실무 절차와 구체적 방법을 중심으로 설명")
        
        # 다양성 요구 시
        if force_diversity:
            instructions.append("7. 기존과 다른 관점이나 접근법을 포함하여 답변")
        
        # 질문 유형별 특화
        if "기관" in question:
            instructions.append("\n**기관 관련 문제**: 기관명, 소속, 근거 법령, 주요 역할을 명확히 제시하세요.")
        elif "특징" in question or "지표" in question:
            instructions.append("\n**특징/지표 문제**: 구체적인 특징과 탐지 방법을 체계적으로 나열하세요.")
        elif "방안" in question or "절차" in question:
            instructions.append("\n**방안/절차 문제**: 단계별 절차와 구체적인 실행 방안을 제시하세요.")
        elif "비율" in question or "기준" in question:
            instructions.append("\n**비율/기준 문제**: 법령에서 정한 정확한 수치와 근거 조항을 명시하세요.")
        
        return "\n".join(instructions)

    def enhance_context_with_examples(self, question: str, question_type: str, domain: str) -> str:
        """예시를 포함한 컨텍스트 강화"""
        
        if question_type == "multiple_choice":
            return self._get_mc_examples(domain)
        else:
            return self._get_subjective_examples(domain)

    def _get_mc_examples(self, domain: str) -> str:
        """객관식 예시"""
        
        examples = {
            "금융투자": {
                "example": "금융투자업 구분에서 소비자금융업은 포함되지 않음",
                "reasoning": "자본시장법상 금융투자업은 6가지로 한정됨"
            },
            "전자금융": {
                "example": "한국은행의 자료제출 요구는 특정 목적으로 제한됨",
                "reasoning": "한국은행법 제91조의 목적 범위 내에서만 가능"
            },
            "개인정보보호": {
                "example": "만 14세 미만 아동은 법정대리인 동의 필요",
                "reasoning": "개인정보보호법 제22조 제6항의 특별 규정"
            }
        }
        
        if domain in examples:
            example = examples[domain]
            return f"\n**유사 예시**: {example['example']}\n**근거**: {example['reasoning']}"
        
        return ""

    def _get_subjective_examples(self, domain: str) -> str:
        """주관식 예시"""
        
        examples = {
            "사이버보안": "트로이 목마의 특징 설명 시 '정상 프로그램 위장 → 시스템 침투 → 악의적 기능 수행' 순서로 설명",
            "전자금융": "분쟁조정 기관 설명 시 '기관명 → 소속 → 근거법령 → 신청방법' 순서로 구성",
            "개인정보보호": "처리 절차 설명 시 '법적 근거 → 동의 요건 → 구체적 절차' 순서로 작성"
        }
        
        if domain in examples:
            return f"\n**답변 구성 예시**: {examples[domain]}"
        
        return ""

    def adapt_for_complexity(self, base_prompt: str, complexity_level: str, question_number: int = None) -> str:
        """복잡도에 따른 프롬프트 적응"""
        
        complexity_adjustments = {
            "초급": {
                "emphasis": "기본 개념과 원칙을 명확하게",
                "detail": "간단명료하게"
            },
            "중급": {
                "emphasis": "실무적 적용과 구체적 절차를",
                "detail": "체계적이고 구체적으로"
            },
            "고급": {
                "emphasis": "법령 조항과 세부 기준을 정밀하게",
                "detail": "전문적이고 정확하게"
            }
        }
        
        if complexity_level in complexity_adjustments:
            adjustment = complexity_adjustments[complexity_level]
            adaptation = f"\n**복잡도 대응**: 이 문제는 {complexity_level} 수준으로 {adjustment['emphasis']} {adjustment['detail']} 접근하세요."
            
            # 후반부 고복잡도 문제 추가 지침
            if question_number is not None and question_number > 300 and complexity_level == "고급":
                adaptation += "\n**후반부 고복잡도**: 예외 조항과 특별 규정도 함께 검토하여 정밀한 답변을 작성하세요."
            
            return base_prompt + adaptation
        
        return base_prompt

    def apply_domain_boost(self, prompt: str, domain: str, question_number: int = None) -> str:
        """도메인 부스트 적용"""
        
        domain_boosts = {
            "사이버보안": "사이버보안 위협의 기술적 특성과 대응 방안을 중심으로",
            "전자금융": "전자금융거래법과 관련 규정의 법적 근거를 중심으로",
            "개인정보보호": "개인정보보호법의 처리 원칙과 권리 보장을 중심으로",
            "정보보안": "정보보안관리체계와 기술적 보안 조치를 중심으로",
            "기타": "관련 법령과 규정의 정확한 해석을 중심으로"
        }
        
        if domain in domain_boosts:
            boost_instruction = f"\n**도메인 특화**: {domain_boosts[domain]} 답변하세요."
            
            # 후반부 기타 도메인 특별 처리
            if question_number is not None and question_number > 300 and domain == "기타":
                boost_instruction += "\n**기타 도메인 후반부**: 구체적 법령 조항과 적용 기준을 명확히 제시하세요."
            
            return prompt + boost_instruction
        
        return prompt

    def _get_fallback_prompt(self, question: str, question_type: str, domain: str) -> str:
        """폴백 프롬프트"""
        
        if question_type == "multiple_choice":
            return f"""다음은 금융보안 관련 객관식 문제입니다. 관련 법령과 규정을 정확히 적용하여 정답을 선택하세요.

**분석 방법**:
1. 질문 유형 확인 (부정형 vs 긍정형)
2. 핵심 키워드 파악
3. 관련 법령 검토
4. 논리적 추론

문제: {question}

정답 번호: """
        else:
            return f"""다음은 금융보안 관련 주관식 문제입니다. 관련 법령을 근거로 정확한 한국어 답변을 작성하세요.

**필수 지침**:
- 모든 답변은 한국어로만 작성
- 관련 법령 조항 명시
- 구체적이고 실무적인 내용 포함

문제: {question}

한국어 답변: """

    def cleanup(self):
        """리소스 정리"""
        pass