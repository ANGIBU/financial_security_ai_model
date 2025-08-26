# prompt_enhancer.py

import random

class PromptEnhancer:
    """프롬프트 구성 및 Few-shot 예시 관리"""
    
    def __init__(self):
        self._initialize_few_shot_examples()
        self._initialize_prompt_templates()
    
    def _initialize_few_shot_examples(self):
        """Few-shot 예시 초기화"""
        
        self.few_shot_examples = {
            "사이버보안": {
                "multiple_choice": [
                    {
                        "question": "다음 중 트로이 목마의 주요 특징으로 가장 적절한 것은?\n1. 자가 복제 기능\n2. 정상 프로그램 위장\n3. 네트워크 속도 저하\n4. 파일 암호화\n5. 화면 잠금",
                        "answer": "2",
                        "reasoning": "트로이 목마는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 것이 주요 특징입니다."
                    },
                    {
                        "question": "SBOM을 금융권에서 활용하는 가장 적절한 이유는?\n1. 데이터 백업\n2. 네트워크 모니터링\n3. 접근 권한 관리\n4. 암호화 강화\n5. 소프트웨어 공급망 보안",
                        "answer": "5",
                        "reasoning": "SBOM은 소프트웨어 구성 요소의 투명성을 제공하여 공급망 보안을 강화하는 목적으로 활용됩니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "트로이 목마 기반 원격제어 악성코드의 특징과 주요 탐지 지표를 설명하세요.",
                        "answer": "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 통해 장기간 시스템에 잠복합니다. 주요 탐지 지표로는 비정상적인 네트워크 통신 패턴, 비인가 프로세스 실행, 파일 시스템 변경, 레지스트리 수정, 시스템 성능 저하 등이 있으며, 실시간 모니터링을 통한 종합적 분석이 필요합니다."
                    }
                ]
            },
            "전자금융": {
                "multiple_choice": [
                    {
                        "question": "한국은행이 금융통화위원회의 요청에 따라 전자금융업자에게 자료제출을 요구할 수 있는 경우로 가장 적절한 것은?\n1. 보안 시스템 점검\n2. 고객 정보 확인\n3. 경영 실적 조사\n4. 통화신용정책 수행\n5. 시장 동향 파악",
                        "answer": "4",
                        "reasoning": "한국은행법에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있습니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "전자금융거래 분쟁조정을 신청할 수 있는 기관을 기술하세요.",
                        "answer": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 전자금융거래법에 근거하여 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하기 위한 업무를 수행합니다."
                    }
                ]
            },
            "개인정보보호": {
                "multiple_choice": [
                    {
                        "question": "만 14세 미만 아동의 개인정보 처리를 위해 필요한 절차로 가장 적절한 것은?\n1. 본인의 직접 동의\n2. 법정대리인의 동의\n3. 학교의 승인\n4. 관할 기관 허가\n5. 보호자 확인서",
                        "answer": "2",
                        "reasoning": "개인정보보호법에 따라 만 14세 미만 아동의 개인정보 처리에는 법정대리인의 동의가 필요합니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "개인정보 침해 신고 및 상담을 담당하는 기관을 기술하세요.",
                        "answer": "개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다. 개인정보보호위원회는 개인정보보호법에 따라 설치된 중앙행정기관으로 개인정보 보호 정책 수립과 감시 업무를 수행합니다."
                    }
                ]
            },
            "금융투자": {
                "multiple_choice": [
                    {
                        "question": "다음 중 금융투자업 구분에 해당하지 않는 것은?\n1. 투자자문업\n2. 투자매매업\n3. 투자중개업\n4. 집합투자업\n5. 소비자금융업",
                        "answer": "5",
                        "reasoning": "소비자금융업은 금융투자업에 해당하지 않으며, 별도의 금융업 분류에 속합니다."
                    }
                ]
            },
            "위험관리": {
                "multiple_choice": [
                    {
                        "question": "위험 관리 계획 수립 시 고려해야 할 요소 중 적절하지 않은 것은?\n1. 수행인력 배정\n2. 위험 수용 정도\n3. 위험 대응 전략\n4. 대상 범위 설정\n5. 수행 기간 설정",
                        "answer": "2",
                        "reasoning": "위험 관리 계획에서 위험 수용은 적절한 관리 요소가 아니며, 위험을 식별하고 대응하는 것이 중요합니다."
                    }
                ]
            },
            "정보보안": {
                "multiple_choice": [
                    {
                        "question": "재해 복구 계획 수립 시 고려 요소 중 옳지 않은 것은?\n1. 복구 절차 수립\n2. 비상연락체계 구축\n3. 개인정보 파기 절차\n4. 복구 목표시간 설정\n5. 백업 시스템 구축",
                        "answer": "3",
                        "reasoning": "개인정보 파기 절차는 재해 복구 계획과 직접적인 관련이 없으며, 복구 관련 요소가 아닙니다."
                    }
                ]
            }
        }
    
    def _initialize_prompt_templates(self):
        """프롬프트 템플릿 초기화"""
        
        self.prompt_templates = {
            "multiple_choice_base": """다음은 금융보안 관련 객관식 문제입니다. 주어진 선택지 중에서 가장 적절한 답을 선택하세요.

{few_shot_examples}

문제: {question}

위 문제를 단계별로 분석하여 정답을 선택하세요.
1. 문제의 핵심 요구사항을 파악
2. 각 선택지를 검토
3. 가장 적절한 답을 선택

정답 번호: """,

            "subjective_base": """다음은 금융보안 관련 주관식 문제입니다. 전문적이고 정확한 답변을 작성하세요.

{few_shot_examples}

참고 정보:
{context_info}

문제: {question}

위 문제에 대해 다음 관점에서 답변하세요:
1. 관련 법령과 규정 근거
2. 구체적인 절차나 방법
3. 실무적 적용 방안

답변: """,

            "institution_question": """다음은 금융보안 관련 기관에 대한 질문입니다.

{few_shot_examples}

기관 정보:
{institution_info}

문제: {question}

위 질문에 대해 정확한 기관명과 근거 법령을 포함하여 답변하세요.

답변: """
        }
    
    def build_few_shot_context(self, domain: str, question_type: str, count: int = 2) -> str:
        """Few-shot 예시 구성"""
        try:
            if domain not in self.few_shot_examples:
                return ""
            
            domain_examples = self.few_shot_examples[domain]
            if question_type not in domain_examples:
                return ""
            
            examples = domain_examples[question_type]
            selected_examples = random.sample(examples, min(count, len(examples)))
            
            few_shot_text = ""
            for i, example in enumerate(selected_examples, 1):
                if question_type == "multiple_choice":
                    few_shot_text += f"예시 {i}:\n문제: {example['question']}\n정답: {example['answer']}\n해설: {example['reasoning']}\n\n"
                else:
                    few_shot_text += f"예시 {i}:\n문제: {example['question']}\n답변: {example['answer']}\n\n"
            
            return few_shot_text
        except Exception as e:
            print(f"Few-shot 컨텍스트 구성 오류: {e}")
            return ""
    
    def build_enhanced_prompt(self, question: str, question_type: str, domain: str = "일반", 
                            context_info: str = "", institution_info: str = "") -> str:
        """프롬프트 구성"""
        try:
            # Few-shot 예시 추가
            few_shot_examples = self.build_few_shot_context(domain, question_type, count=2)
            
            # 기관 질문 특별 처리
            if "기관" in question.lower() and institution_info:
                template = self.prompt_templates["institution_question"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    institution_info=institution_info,
                    question=question
                )
            
            # 일반 프롬프트
            if question_type == "multiple_choice":
                template = self.prompt_templates["multiple_choice_base"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            else:
                template = self.prompt_templates["subjective_base"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    context_info=context_info if context_info else "해당 도메인의 관련 법령과 규정을 참고하세요.",
                    question=question
                )
                
        except Exception as e:
            print(f"프롬프트 구성 오류: {e}")
            # 기본 프롬프트 반환
            return f"문제: {question}\n\n위 문제에 대해 전문적이고 정확한 답변을 작성하세요.\n\n답변: "
    
    def get_context_hints(self, domain: str, intent_type: str) -> str:
        """도메인별 컨텍스트 힌트 제공"""
        
        context_hints = {
            "사이버보안": {
                "특징_묻기": "사이버 위협의 기술적 특성과 동작 방식을 중심으로 설명하세요.",
                "지표_묻기": "탐지 가능한 기술적 징후와 모니터링 방법을 포함하여 설명하세요.",
                "방안_묻기": "다층 방어체계와 실시간 대응 방안을 중심으로 설명하세요."
            },
            "전자금융": {
                "기관_묻기": "전자금융거래법과 관련 기관의 역할을 명확히 설명하세요.",
                "방안_묻기": "전자금융거래의 안전성과 이용자 보호 방안을 중심으로 설명하세요."
            },
            "개인정보보호": {
                "기관_묻기": "개인정보보호법과 관련 기관의 업무를 구체적으로 설명하세요.",
                "방안_묻기": "개인정보 처리 원칙과 정보주체 권리 보장 방안을 중심으로 설명하세요."
            },
            "금융투자": {
                "방안_묻기": "자본시장법과 투자자 보호 원칙을 중심으로 설명하세요."
            },
            "위험관리": {
                "방안_묻기": "위험 식별, 평가, 대응, 모니터링의 단계별 절차를 포함하여 설명하세요."
            },
            "정보보안": {
                "방안_묻기": "정보보안관리체계의 수립과 운영 절차를 중심으로 설명하세요."
            }
        }
        
        try:
            return context_hints.get(domain, {}).get(intent_type, "관련 법령과 실무 적용 방안을 포함하여 설명하세요.")
        except Exception:
            return "관련 법령과 실무 적용 방안을 포함하여 설명하세요."
    
    def cleanup(self):
        """리소스 정리"""
        pass