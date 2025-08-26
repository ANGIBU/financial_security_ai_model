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
                        "question": "다음 중 트로이 목마의 주요 특징으로 가장 적절한 것은?\n1 자가 복제 기능\n2 정상 프로그램 위장\n3 네트워크 속도 저하\n4 파일 암호화\n5 화면 잠금",
                        "answer": "2",
                        "reasoning": "트로이 목마는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 것이 주요 특징입니다."
                    },
                    {
                        "question": "SBOM을 금융권에서 활용하는 가장 적절한 이유는?\n1 데이터 백업\n2 네트워크 모니터링\n3 접근 권한 관리\n4 암호화 강화\n5 소프트웨어 공급망 보안",
                        "answer": "5",
                        "reasoning": "SBOM은 소프트웨어 구성 요소의 투명성을 제공하여 공급망 보안을 강화하는 목적으로 활용됩니다."
                    },
                    {
                        "question": "딥페이크 기술의 악용을 방지하기 위한 금융권의 선제적 대응 방안으로 가장 적절한 것은?\n1 딥페이크 탐지 기능이 없는 구식 인증 시스템 도입\n2 딥보이스 탐지 기술 개발\n3 금융기관의 음성 복제\n4 딥페이크 영상 제작 지원\n5 금융소비자 홍보 강화",
                        "answer": "2",
                        "reasoning": "딥페이크 기술 악용 방지를 위한 선제적 대응 방안으로는 딥보이스 탐지 기술 개발이 가장 적절합니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요.",
                        "answer": "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하고 외부에서 원격으로 제어하는 특성을 가집니다. 주요 탐지 지표로는 비정상적인 네트워크 통신 패턴, 비인가 프로세스 실행, 파일 시스템 변경 등이 있으며 실시간 모니터링을 통한 종합적 분석이 필요합니다."
                    },
                    {
                        "question": "딥페이크 기술 악용에 대비한 금융권의 대응 방안을 기술하세요.",
                        "answer": "딥페이크 기술 악용에 대비하여 다층 방어체계 구축, 실시간 탐지 시스템 도입, 생체인증과 다중 인증 체계를 통한 신원 검증 강화, 직원 교육 및 인식 제고를 통한 종합적 보안 대응방안이 필요합니다."
                    },
                    {
                        "question": "디지털 지갑(Digital Wallet)에서 우려되는 주요 보안 위협을 설명하세요.",
                        "answer": "디지털 지갑의 주요 보안 위협으로는 개인키 도난, 피싱 공격, 멀웨어 감염, 스마트 컨트랙트 취약점이 있으며 다중 인증과 하드웨어 지갑 사용이 권장됩니다."
                    }
                ]
            },
            "전자금융": {
                "multiple_choice": [
                    {
                        "question": "한국은행이 금융통화위원회의 요청에 따라 전자금융업자에게 자료제출을 요구할 수 있는 경우로 가장 적절한 것은?\n1 보안 시스템 점검\n2 고객 정보 확인\n3 경영 실적 조사\n4 통화신용정책 수행\n5 시장 동향 파악",
                        "answer": "4",
                        "reasoning": "한국은행법에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있습니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요.",
                        "answer": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당하며, 금융감독원 내에 설치되어 전자금융거래법에 근거하여 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결합니다."
                    },
                    {
                        "question": "금융회사가 정보보호 예산을 관리할 때, 전자금융감독규정상 정보기술부문 인력 및 예산의 기준 비율은 얼마인가요?",
                        "answer": "전자금융감독규정 제16조에 따라 금융회사는 정보기술부문 인력을 총 인력의 5% 이상, 정보기술부문 예산을 총 예산의 7% 이상 정보보호 업무에 배정해야 합니다. 다만 회사 규모, 업무 특성, 정보기술 위험수준 등에 따라 금융감독원장이 별도로 정할 수 있습니다."
                    }
                ]
            },
            "개인정보보호": {
                "multiple_choice": [
                    {
                        "question": "만 14세 미만 아동의 개인정보 처리를 위해 필요한 절차로 가장 적절한 것은?\n1 본인의 직접 동의\n2 법정대리인의 동의\n3 학교의 승인\n4 관할 기관 허가\n5 보호자 확인서",
                        "answer": "2",
                        "reasoning": "개인정보보호법에 따라 만 14세 미만 아동의 개인정보 처리에는 법정대리인의 동의가 필요합니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "개인정보 접근 권한 검토는 어떻게 수행해야 하며, 그 목적은 무엇인가요?",
                        "answer": "개인정보 접근 권한 검토는 업무상 필요한 최소한의 권한만을 부여하는 최소권한 원칙에 따라 정기적으로 수행하며, 불필요한 권한은 즉시 회수하여 개인정보 오남용을 방지해야 합니다."
                    }
                ]
            },
            "정보보안": {
                "multiple_choice": [
                    {
                        "question": "재해 복구 계획 수립 시 고려 요소 중 옳지 않은 것은?\n1 복구 절차 수립\n2 비상연락체계 구축\n3 개인정보 파기 절차\n4 복구 목표시간 설정\n5 백업 시스템 구축",
                        "answer": "3",
                        "reasoning": "개인정보 파기 절차는 재해 복구 계획과 직접적인 관련이 없으며, 복구 관련 요소가 아닙니다."
                    }
                ],
                "subjective": [
                    {
                        "question": "정보보호의 3대 요소에 해당하는 보안 목표를 3가지 기술하세요.",
                        "answer": "정보보호의 3대 요소는 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)으로 구성되며, 이를 통해 정보자산의 안전한 보호와 관리를 보장합니다."
                    },
                    {
                        "question": "SMTP 프로토콜의 보안상 주요 역할을 설명하세요.",
                        "answer": "SMTP 프로토콜은 이메일 전송을 담당하며, 보안상 주요 역할로는 인증 메커니즘 제공, 암호화 통신 지원, 스팸 및 악성 이메일 차단을 통해 안전한 이메일 서비스를 보장합니다."
                    }
                ]
            },
            "정보통신": {
                "multiple_choice": [
                    {
                        "question": "집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 과학기술정보통신부장관에게 보고해야 하는 사항으로 옳지 않은 것은?\n1 정보통신서비스 제공의 중단이 발생한 일시 및 장소\n2 정보통신서비스 제공의 중단이 발생한 원인에 대한 법적 책임\n3 정보통신서비스 제공의 중단이 발생한 원인 및 피해내용\n4 응급조치 사항",
                        "answer": "2",
                        "reasoning": "정보통신서비스 제공 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항에는 법적 책임이 포함되지 않습니다."
                    }
                ]
            },
            "금융투자": {
                "multiple_choice": [
                    {
                        "question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업",
                        "answer": "1",
                        "reasoning": "소비자금융업은 금융투자업에 해당하지 않으며, 별도의 금융업 분류에 속합니다."
                    }
                ]
            },
            "위험관리": {
                "multiple_choice": [
                    {
                        "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간",
                        "answer": "2",
                        "reasoning": "위험 관리 계획에서 위험 수용은 적절한 관리 요소가 아니며, 위험을 식별하고 대응하는 것이 중요합니다."
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

위 문제를 다음 단계로 분석하여 정답을 선택하세요:
1. 문제의 핵심 키워드와 요구사항 파악
2. 각 선택지를 해당 법령과 규정에 따라 검토
3. 문제 유형(부정/긍정)에 따른 논리적 추론
4. 전문가 관점에서 최적의 답안 선택

정답 번호: """,

            "subjective_base": """다음은 금융보안 관련 주관식 문제입니다. 반드시 한국어로만 전문적이고 정확한 답변을 작성하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성
- 관련 법령과 규정에 근거한 전문적 답변 작성
- 구체적이고 실무적인 내용 포함
- 자연스러운 한국어 문장으로 구성

{few_shot_examples}

참고 정보:
{context_info}

문제: {question}

위 문제에 대해 관련 법령과 규정을 근거로 구체적이고 전문적인 한국어 답변을 작성하세요.

한국어 답변: """,

            "institution_question": """다음은 금융보안 관련 기관에 대한 질문입니다. 반드시 한국어로만 답변하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성
- 기관의 정확한 명칭과 역할 기술

{few_shot_examples}

기관 정보:
{institution_info}

문제: {question}

위 질문에 대해 다음 요소를 포함하여 한국어로 답변하세요:
1. 정확한 기관명과 소속 조직
2. 법적 근거와 설립 배경
3. 주요 업무와 권한 범위
4. 관련 절차와 연락 방법

한국어 답변: """,

            "ratio_question": """다음은 금융보안 관련 비율에 대한 질문입니다. 반드시 구체적인 수치와 법적 근거를 포함하여 한국어로만 답변하세요.

답변 작성 지침:
- 정확한 수치와 퍼센트 명시
- 해당 법령과 조항 인용
- 예외 조건이나 특별 규정 포함

{few_shot_examples}

문제: {question}

위 질문에 대해 다음 사항을 포함하여 한국어로 답변하세요:
1. 정확한 비율과 수치
2. 관련 법령과 조항 번호
3. 적용 조건 및 예외사항
4. 감독기관의 재량권

한국어 답변: """,

            "domain_specific": {
                "사이버보안": """다음은 사이버보안 관련 문제입니다. 기술적 특성과 보안 대응 방안에 중점을 두어 답변하세요.

{few_shot_examples}

문제: {question}

사이버보안 관점에서 기술적 특징, 탐지 방법, 대응 방안을 구체적으로 설명하세요.

한국어 답변: """,

                "전자금융": """다음은 전자금융 관련 문제입니다. 전자금융거래법과 관련 규정을 근거로 답변하세요.

{few_shot_examples}

문제: {question}

전자금융거래법에 근거하여 법적 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """,

                "개인정보보호": """다음은 개인정보보호 관련 문제입니다. 개인정보보호법을 근거로 답변하세요.

{few_shot_examples}

문제: {question}

개인정보보호법에 따른 처리 원칙과 절차를 구체적으로 설명하세요.

한국어 답변: """,

                "정보보안": """다음은 정보보안 관련 문제입니다. 정보보안관리체계 관점에서 답변하세요.

{few_shot_examples}

문제: {question}

정보보안관리체계 구축과 운영 관점에서 체계적으로 설명하세요.

한국어 답변: """,

                "정보통신": """다음은 정보통신 관련 문제입니다. 정보통신기반 보호법을 근거로 답변하세요.

{few_shot_examples}

문제: {question}

정보통신기반 보호법에 따른 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """
            }
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
            
            if question_type == "subjective":
                count = min(count, 1)
            else:
                count = min(count, len(examples))
            
            selected_examples = random.sample(examples, count) if len(examples) > count else examples
            
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
            example_count = 2 if domain in ["개인정보보호", "전자금융"] else 1
            few_shot_examples = self.build_few_shot_context(domain, question_type, count=example_count)
            
            # 비율 관련 질문 특별 처리
            if self._is_ratio_question(question, domain):
                template = self.prompt_templates["ratio_question"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 기관 질문 특별 처리
            if ("기관" in question.lower() or "위원회" in question.lower()) and institution_info:
                template = self.prompt_templates["institution_question"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    institution_info=institution_info,
                    question=question
                )
            
            # 도메인별 특화 템플릿 사용
            if domain in self.prompt_templates["domain_specific"] and question_type == "subjective":
                template = self.prompt_templates["domain_specific"][domain]
                return template.format(
                    few_shot_examples=few_shot_examples,
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
                simplified_context = context_info[:500] + "..." if len(context_info) > 500 else context_info
                return template.format(
                    few_shot_examples=few_shot_examples,
                    context_info=simplified_context if simplified_context else "관련 법령과 규정을 참고하세요.",
                    question=question
                )
                
        except Exception as e:
            print(f"프롬프트 구성 오류: {e}")
            if question_type == "multiple_choice":
                return f"""다음 문제의 정답 번호를 선택하세요.

문제: {question}

정답 번호: """
            else:
                return f"""다음 문제에 대해 한국어로만 전문적인 답변을 작성하세요.

문제: {question}

한국어 답변: """

    def _is_ratio_question(self, question: str, domain: str) -> bool:
        """비율 관련 질문 확인"""
        question_lower = question.lower()
        
        ratio_indicators = [
            "비율", "얼마", "기준", "퍼센트", "%", 
            "정보기술부문", "인력", "예산", "배정"
        ]
        
        # 전자금융 도메인에서 정보기술부문 관련 질문
        if domain == "전자금융":
            if any(indicator in question_lower for indicator in ratio_indicators):
                if "정보기술부문" in question_lower or "예산" in question_lower:
                    return True
        
        # 일반적인 비율 질문
        ratio_count = sum(1 for indicator in ratio_indicators if indicator in question_lower)
        return ratio_count >= 2
    
    def get_context_hints(self, domain: str, intent_type: str) -> str:
        """도메인별 컨텍스트 힌트 제공"""
        
        context_hints = {
            "사이버보안": {
                "특징_묻기": "사이버 위협의 기술적 특성과 동작 방식, 은밀성과 지속성을 중심으로 설명하세요.",
                "지표_묻기": "네트워크 트래픽, 프로세스 활동, 파일 시스템 변화 등 구체적인 탐지 지표를 포함하여 설명하세요.",
                "방안_묻기": "다층 방어체계, 실시간 모니터링, 사고 대응 절차를 포함한 종합적 방안을 설명하세요."
            },
            "전자금융": {
                "기관_묻기": "전자금융거래법에 근거한 기관의 법적 지위와 구체적 업무 범위를 명확히 설명하세요.",
                "방안_묻기": "접근매체 보안, 거래 기록 보존, 분쟁조정 절차를 포함한 이용자 보호 방안을 설명하세요.",
                "절차_묻기": "전자금융거래법에 명시된 법적 절차와 당사자별 의무사항을 단계별로 설명하세요.",
                "비율_묻기": "전자금융감독규정에 명시된 구체적인 수치와 법적 근거를 포함하여 설명하세요."
            },
            "개인정보보호": {
                "기관_묻기": "개인정보보호법에 따른 기관의 권한과 개인정보 처리 감독 업무를 구체적으로 설명하세요.",
                "방안_묻기": "수집 최소화, 목적 제한, 정보주체 권리 보장 원칙을 중심으로 설명하세요.",
                "절차_묻기": "동의 획득, 처리 현황 공개, 권리 행사 절차를 법령에 따라 설명하세요."
            },
            "금융투자": {
                "방안_묻기": "자본시장법의 투자자 보호 원칙과 적합성 원칙 적용 방안을 중심으로 설명하세요."
            },
            "위험관리": {
                "방안_묻기": "위험 식별, 평가, 대응, 모니터링의 4단계 절차와 각 단계별 핵심 활동을 설명하세요."
            },
            "정보보안": {
                "방안_묻기": "정보보안관리체계의 수립, 운영, 점검, 개선 사이클을 중심으로 설명하세요."
            },
            "정보통신": {
                "방안_묻기": "정보통신기반 보호법에 따른 보고 요구사항과 절차를 명확히 설명하세요."
            }
        }
        
        try:
            return context_hints.get(domain, {}).get(intent_type, "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 체계적으로 설명하세요.")
        except Exception:
            return "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 체계적으로 설명하세요."
    
    def cleanup(self):
        """리소스 정리"""
        pass