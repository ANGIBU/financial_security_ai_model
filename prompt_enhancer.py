# prompt_enhancer.py

import re
import random
import hashlib
from typing import Dict, List


class PromptEnhancer:
    """프롬프트 구성 및 Few-shot 예시 관리 - 정확도 향상 버전"""
    
    def __init__(self):
        self._initialize_enhanced_few_shot_examples()
        self._initialize_improved_prompt_templates()
        self.used_examples_cache = {}
        self.prompt_history = []
        
    def _initialize_enhanced_few_shot_examples(self):
        """향상된 Few-shot 예시 초기화 - 정확도 중심"""
        
        self.few_shot_examples = {
            "사이버보안": {
                "multiple_choice": [
                    {
                        "question": "다음 중 트로이 목마 기반 원격제어 악성코드(RAT)의 주요 특징으로 가장 적절한 것은?\n1 자동 복제 및 확산\n2 정상 프로그램으로 위장하여 침투\n3 시스템 파일 암호화\n4 네트워크 트래픽 차단\n5 하드웨어 손상 유발",
                        "answer": "2",
                        "reasoning": "트로이 목마의 핵심 특징은 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 것입니다. RAT는 이러한 위장을 통해 원격제어 기능을 숨깁니다.",
                        "domain_focus": ["트로이", "RAT", "특징", "위장"],
                        "difficulty": "중급"
                    },
                    {
                        "question": "SBOM을 금융권에서 활용하는 주된 목적으로 가장 적절한 것은?\n1 시스템 성능 최적화\n2 네트워크 대역폭 관리\n3 사용자 인증 강화\n4 암호화 키 관리\n5 소프트웨어 공급망 보안 강화",
                        "answer": "5",
                        "reasoning": "SBOM(Software Bill of Materials)은 소프트웨어 구성 요소의 투명성을 제공하여 공급망 공격을 예방하고 취약점을 효율적으로 관리하는 것이 주요 목적입니다.",
                        "domain_focus": ["SBOM", "공급망", "보안", "활용"],
                        "difficulty": "고급"
                    },
                    {
                        "question": "딥페이크 기술 악용에 대한 금융권의 선제적 대응 방안으로 가장 적절한 것은?\n1 기존 CCTV 시스템 확대\n2 딥보이스 탐지 기술 도입\n3 고객 신분증 확인 강화\n4 콜센터 운영 시간 단축\n5 온라인 서비스 전면 중단",
                        "answer": "2",
                        "reasoning": "딥페이크 음성 기술에 대응하려면 딥보이스 탐지 기술을 도입하여 실시간으로 가짜 음성을 구별하는 것이 가장 효과적인 선제적 대응 방안입니다.",
                        "domain_focus": ["딥페이크", "선제적", "대응", "탐지"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요.",
                        "answer": "트로이 목마 기반 원격제어 악성코드는 정상적인 응용프로그램으로 위장하여 시스템에 침투한 후, 외부 공격자가 감염된 시스템을 원격으로 제어할 수 있게 하는 악성코드입니다. 주요 특징으로는 은밀한 설치, 지속적인 통신, 관리자 권한 획득 등이 있습니다. 주요 탐지 지표로는 비정상적인 네트워크 외부 통신, 알 수 없는 프로세스의 지속적 실행, 파일 시스템의 무단 변경, 레지스트리 키 수정, CPU 사용률 이상 증가 등이 있으며, 행위 기반 분석과 네트워크 트래픽 모니터링을 통한 종합적 탐지가 필요합니다.",
                        "domain_focus": ["트로이", "RAT", "특징", "탐지지표"],
                        "answer_type": "특징설명",
                        "quality_score": 0.95
                    }
                ]
            },
            "전자금융": {
                "multiple_choice": [
                    {
                        "question": "한국은행이 금융통화위원회의 요청에 따라 전자금융업자에게 자료제출을 요구할 수 있는 경우로 가장 적절한 것은?\n1 금융기관 경영 실태 조사\n2 고객 정보 보안 점검\n3 시장 경쟁력 분석\n4 통화신용정책 수행 및 지급결제제도 운영\n5 세무 조사 지원",
                        "answer": "4",
                        "reasoning": "한국은행법 제91조에 따르면, 한국은행은 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 필요한 경우에만 금융기관에 자료제출을 요구할 수 있습니다.",
                        "domain_focus": ["한국은행", "자료제출", "통화신용정책", "지급결제"],
                        "difficulty": "고급"
                    },
                    {
                        "question": "전자금융감독규정에 따른 금융회사의 정보보호 예산 배정 기준으로 올바른 것은?\n1 정보기술부문 인력 3% 이상, 예산 5% 이상\n2 정보기술부문 인력 5% 이상, 예산 7% 이상\n3 정보기술부문 인력 7% 이상, 예산 10% 이상\n4 정보기술부문 인력 10% 이상, 예산 15% 이상\n5 별도 기준 없음",
                        "answer": "2",
                        "reasoning": "전자금융감독규정 제16조에 따르면, 금융회사는 정보기술부문 인력을 총 인력의 5% 이상, 정보기술부문 예산을 총 예산의 7% 이상을 정보보호 업무에 배정해야 합니다.",
                        "domain_focus": ["정보기술부문", "인력", "예산", "5%", "7%"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "전자금융거래법에 따라 이용자가 분쟁조정을 신청할 수 있는 기관을 기술하세요.",
                        "answer": "전자금융거래법 제28조에 따라 이용자는 전자금융분쟁조정위원회에 분쟁조정을 신청할 수 있습니다. 전자금융분쟁조정위원회는 금융감독원 내에 설치되어 있으며, 전자금융거래와 관련하여 이용자와 전자금융업자 간에 발생한 분쟁을 공정하고 신속하게 해결하는 역할을 수행합니다. 이용자는 온라인 또는 서면을 통해 분쟁조정을 신청할 수 있으며, 조정 과정은 무료로 진행됩니다.",
                        "domain_focus": ["전자금융분쟁조정위원회", "분쟁조정", "신청", "기관"],
                        "answer_type": "기관명",
                        "quality_score": 0.95
                    }
                ]
            },
            "개인정보보호": {
                "multiple_choice": [
                    {
                        "question": "개인정보보호법에 따라 만 14세 미만 아동의 개인정보 처리를 위해 필요한 절차는?\n1 아동 본인의 서면 동의\n2 법정대리인의 동의\n3 학교장의 승인\n4 관할 지방자치단체의 허가\n5 개인정보보호위원회의 승인",
                        "answer": "2",
                        "reasoning": "개인정보보호법 제22조 제6항에 따르면, 만 14세 미만 아동의 개인정보를 처리하려면 법정대리인의 동의를 받아야 합니다. 이는 아동의 개인정보 자기결정권을 보호하기 위한 규정입니다.",
                        "domain_focus": ["만14세", "아동", "법정대리인", "동의"],
                        "difficulty": "중급"
                    },
                    {
                        "question": "개인정보 관리체계 수립 시 정책 수립 단계에서 가장 중요한 요소는?\n1 개인정보보호 담당자 지정\n2 경영진의 의지와 참여\n3 정보보호 시스템 구축\n4 직원 교육 프로그램 운영\n5 개인정보영향평가 실시",
                        "answer": "2",
                        "reasoning": "개인정보 관리체계 수립에서 정책 수립 단계의 핵심은 최고경영진의 개인정보보호에 대한 확고한 의지와 적극적인 참여입니다. 이는 체계적이고 효과적인 관리체계 운영의 기반이 됩니다.",
                        "domain_focus": ["관리체계", "정책수립", "경영진", "중요한요소"],
                        "difficulty": "중급"
                    }
                ],
                "subjective": [
                    {
                        "question": "개인정보보호위원회에서 담당하는 주요 업무와 개인정보 침해신고를 접수하는 기관을 설명하세요.",
                        "answer": "개인정보보호위원회는 개인정보보호법 제7조에 따라 설치된 국무총리 소속의 중앙행정기관으로, 개인정보 보호에 관한 정책의 수립 및 시행, 개인정보 처리 실태 조사, 개인정보보호 교육 및 홍보 등의 업무를 총괄합니다. 개인정보 침해신고는 개인정보보호위원회에서 운영하는 개인정보침해신고센터(privacy.go.kr)에서 접수하며, 온라인 신고, 전화 상담(국번 없이 182), 방문 상담을 통해 개인정보 침해신고 및 상담 업무를 수행합니다.",
                        "domain_focus": ["개인정보보호위원회", "침해신고센터", "업무", "신고"],
                        "answer_type": "기관명",
                        "quality_score": 0.92
                    }
                ]
            },
            "정보보안": {
                "multiple_choice": [
                    {
                        "question": "정보보호의 3대 요소에 해당하는 것은?\n1 기밀성, 인증성, 부인방지\n2 기밀성, 무결성, 가용성\n3 무결성, 인증성, 접근통제\n4 가용성, 부인방지, 감사성\n5 접근통제, 암호화, 백업",
                        "answer": "2",
                        "reasoning": "정보보호의 3대 요소는 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)입니다. 이를 CIA 트라이어드라고 하며, 정보보안의 기본 목표를 나타냅니다.",
                        "domain_focus": ["3대요소", "기밀성", "무결성", "가용성"],
                        "difficulty": "초급"
                    },
                    {
                        "question": "재해복구 계획 수립 시 고려해야 할 요소 중 옳지 않은 것은?\n1 복구 절차 수립\n2 비상연락체계 구축\n3 개인정보 파기 절차\n4 복구 목표시간(RTO) 설정\n5 백업 시스템 운영 방안",
                        "answer": "3",
                        "reasoning": "재해복구 계획에는 복구 절차, 비상연락체계, RTO 설정, 백업 시스템 등이 포함되지만, 개인정보 파기 절차는 재해복구와 직접적인 관련이 없는 별개의 업무입니다.",
                        "domain_focus": ["재해복구", "계획수립", "개인정보파기", "옳지않은"],
                        "difficulty": "중급"
                    }
                ],
                "subjective": [
                    {
                        "question": "정보보호의 3대 요소를 정의하고 각각의 의미를 설명하세요.",
                        "answer": "정보보호의 3대 요소는 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)입니다. 기밀성은 인가된 사용자만이 정보에 접근할 수 있도록 하여 정보의 노출을 방지하는 것을 의미합니다. 무결성은 정보가 무단으로 변경, 삭제, 생성되지 않도록 하여 정보의 정확성과 완전성을 보장하는 것입니다. 가용성은 인가된 사용자가 필요할 때 언제든지 정보와 관련 자원에 접근할 수 있도록 시스템의 지속적인 운영을 보장하는 것을 말합니다.",
                        "domain_focus": ["3대요소", "기밀성", "무결성", "가용성", "정의"],
                        "answer_type": "정의설명",
                        "quality_score": 0.93
                    }
                ]
            },
            "정보통신": {
                "multiple_choice": [
                    {
                        "question": "정보통신서비스 제공의 중단이 발생했을 때 과학기술정보통신부장관에게 보고해야 하는 사항으로 옳지 않은 것은?\n1 정보통신서비스 제공의 중단이 발생한 일시 및 장소\n2 정보통신서비스 제공의 중단이 발생한 원인에 대한 법적 책임\n3 정보통신서비스 제공의 중단이 발생한 원인 및 피해내용\n4 응급조치 사항\n5 복구 예상 시간",
                        "answer": "2",
                        "reasoning": "정보통신기반 보호법에 따른 보고 사항에는 중단 발생 일시 및 장소, 원인 및 피해내용, 응급조치 사항이 포함되지만, 법적 책임에 관한 사항은 보고 대상에 해당하지 않습니다.",
                        "domain_focus": ["정보통신서비스", "중단", "보고사항", "법적책임", "옳지않은"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "정보통신시설의 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항을 설명하세요.",
                        "answer": "정보통신기반 보호법에 따라 집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 과학기술정보통신부장관에게 보고해야 하는 사항은 다음과 같습니다. 첫째, 정보통신서비스 제공의 중단이 발생한 일시 및 장소, 둘째, 정보통신서비스 제공의 중단이 발생한 원인 및 피해내용, 셋째, 응급조치 사항입니다. 다만, 법적 책임에 관한 사항은 보고 대상에 해당하지 않습니다.",
                        "domain_focus": ["정보통신시설", "중단", "보고사항", "과학기술정보통신부"],
                        "answer_type": "절차설명",
                        "quality_score": 0.90
                    }
                ]
            },
            "금융투자": {
                "multiple_choice": [
                    {
                        "question": "자본시장법상 금융투자업의 구분에 해당하지 않는 것은?\n1 투자자문업\n2 투자매매업\n3 투자중개업\n4 집합투자업\n5 소비자금융업",
                        "answer": "5",
                        "reasoning": "자본시장법에 따른 금융투자업은 투자자문업, 투자매매업, 투자중개업, 집합투자업, 신탁업, 투자일임업 등으로 구분되며, 소비자금융업은 금융투자업에 해당하지 않습니다.",
                        "domain_focus": ["금융투자업", "구분", "소비자금융업", "해당하지않는"],
                        "difficulty": "중급"
                    }
                ]
            },
            "위험관리": {
                "multiple_choice": [
                    {
                        "question": "위험관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 위험관리 수행인력\n2 위험 수용 정책\n3 위험 대응 전략 선정\n4 위험관리 대상 범위\n5 위험관리 수행 기간",
                        "answer": "2",
                        "reasoning": "위험관리 계획에서는 위험을 식별하고 적극적으로 대응하는 것이 중요하며, 단순히 위험을 수용하는 것보다는 위험 회피, 위험 감소, 위험 전가 등의 적극적 대응 전략을 수립해야 합니다.",
                        "domain_focus": ["위험관리", "계획수립", "위험수용", "적절하지않은"],
                        "difficulty": "중급"
                    }
                ]
            }
        }
    
    def _initialize_improved_prompt_templates(self):
        """개선된 프롬프트 템플릿 초기화 - Chain of Thought 강화"""
        
        self.prompt_templates = {
            "multiple_choice_enhanced": """다음은 금융보안 관련 객관식 문제입니다. 체계적 분석을 통해 정확한 답을 선택하세요.

{few_shot_examples}

**단계별 분석 방법:**
1단계: 질문의 핵심 키워드와 요구사항 파악
2단계: 문제 유형 판단 (긍정형 vs 부정형)
3단계: 각 선택지를 관련 법령과 규정에 따라 검토
4단계: 논리적 추론을 통한 정답 도출

**중요 지침:**
- "해당하지 않는", "적절하지 않은", "옳지 않은" → 조건에 맞지 않는 선택지 찾기
- "가장 적절한", "가장 옳은", "맞는 것" → 조건에 가장 부합하는 선택지 선택
- 법령과 규정의 정확한 조항과 기준을 적용
- 도메인별 전문 지식과 실무 관점 고려

문제: {question}

위 4단계 분석 방법에 따라 체계적으로 검토한 후, 정답 번호만 제시하세요.

정답 번호: """,

            "subjective_enhanced": """다음은 금융보안 관련 주관식 문제입니다. 다음 작성 지침에 따라 정확하고 전문적인 한국어 답변을 작성하세요.

**필수 작성 지침:**
1. 모든 답변은 한국어로만 작성 (영어 사용 절대 금지)
2. 관련 법령의 구체적 조항과 근거 명시
3. 실무적이고 구체적인 내용 포함
4. 자연스러운 한국어 문장으로 구성
5. 전문용어의 정확한 사용과 설명

{few_shot_examples}

**도메인별 참고 정보:**
{context_info}

**답변 구성 요소:**
- 핵심 개념의 정의 또는 기관/제도 소개
- 관련 법령과 구체적 조항 인용
- 주요 특징, 절차, 기준 등의 상세 설명
- 실무 적용 방법이나 주의사항

문제: {question}

위 지침에 따라 관련 법령과 규정을 정확히 인용하면서 구체적이고 전문적인 한국어 답변을 작성하세요.

한국어 답변: """,

            "institution_specialized": """다음은 금융보안 관련 기관에 대한 질문입니다. 정확한 기관 정보를 바탕으로 한국어로만 답변하세요.

**기관 질문 답변 지침:**
1. 정확한 기관명과 공식 명칭 사용
2. 설치 근거 법령과 조항 명시
3. 소속 기관과 조직 체계 설명
4. 구체적인 권한과 업무 범위 기술
5. 관련 절차와 연락 방법 포함

{few_shot_examples}

**기관 정보:**
{institution_info}

**답변 필수 포함 요소:**
- 정확한 기관명과 소속 조직
- 설치 근거 법령 및 조항
- 주요 권한과 업무 범위
- 관련 절차 및 신청 방법
- 연락처 및 접근 방법

문제: {question}

위 요소들을 모두 포함하여 정확하고 상세한 한국어 답변을 작성하세요.

한국어 답변: """,

            "ratio_specialized": """다음은 금융보안 관련 비율 및 수치에 대한 질문입니다. 정확한 수치와 법적 근거를 포함하여 답변하세요.

**비율 질문 답변 지침:**
1. 정확한 수치와 단위(%, 비율, 기준) 명시
2. 관련 법령과 구체적 조항 번호 인용
3. 적용 대상과 범위 명확히 설명
4. 예외 조건이나 특별 규정 포함
5. 감독기관의 재량권과 기준 설명

{few_shot_examples}

**답변 필수 포함 요소:**
- 정확한 수치와 기준 (예: 5% 이상, 7% 이상)
- 관련 법령명과 조항 번호
- 적용 조건 및 대상
- 예외사항 및 특별규정
- 감독기관의 권한과 기준

문제: {question}

위 지침에 따라 정확한 수치와 법적 근거를 포함한 상세한 한국어 답변을 작성하세요.

한국어 답변: """,

            "domain_specialized": {
                "사이버보안": """다음은 사이버보안 관련 문제입니다. 기술적 메커니즘과 실무적 대응 방안을 중심으로 답변하세요.

**사이버보안 답변 지침:**
- 위협의 기술적 동작 원리와 메커니즘 상세 설명
- 실제 공격 사례와 구체적인 탐지 지표 제시
- 다층 방어체계와 종합적 대응 방안 포함
- 최신 보안 기술과 표준 적용 방법 설명

{few_shot_examples}

문제: {question}

사이버보안 전문가 관점에서 기술적 특징, 탐지 방법, 종합적 대응 방안을 구체적으로 설명하세요.

한국어 답변: """,

                "전자금융": """다음은 전자금융 관련 문제입니다. 전자금융거래법과 관련 규정의 정확한 적용을 중심으로 답변하세요.

**전자금융 답변 지침:**
- 전자금융거래법의 구체적 조항과 번호 인용
- 이용자 보호 조치와 업무 절차 중심 설명
- 감독기관의 권한과 업무 범위 명확히 제시
- 실무에서의 적용 방법과 준수 사항 포함

{few_shot_examples}

문제: {question}

전자금융거래법과 전자금융감독규정의 정확한 조항에 근거하여 법적 요구사항과 실무 절차를 명확히 설명하세요.

한국어 답변: """,

                "개인정보보호": """다음은 개인정보보호 관련 문제입니다. 개인정보보호법의 원칙과 절차를 정확히 적용하여 답변하세요.

**개인정보보호 답변 지침:**
- 개인정보보호법의 기본 원칙과 구체적 조항 인용
- 정보주체의 권리와 처리자의 의무 명확히 구분
- 동의, 수집, 이용, 제공, 파기 등 처리 단계별 요구사항
- 실무에서의 적용 사례와 주의사항 포함

{few_shot_examples}

문제: {question}

개인정보보호법의 처리 원칙과 정보주체 권리를 중심으로 법적 요구사항과 실무 절차를 구체적으로 설명하세요.

한국어 답변: """,

                "정보보안": """다음은 정보보안 관련 문제입니다. 정보보안관리체계와 보안 통제 관점에서 답변하세요.

**정보보안 답변 지침:**
- 정보보안관리체계(ISMS)의 체계적 접근 방법 적용
- 기술적, 관리적, 물리적 보안대책의 구분과 연계성
- 위험 분석과 단계별 대응 방안의 체계적 제시
- 보안 정책, 절차, 가이드라인의 구체적 내용 포함

{few_shot_examples}

문제: {question}

정보보안관리체계의 구축과 운영 관점에서 체계적인 보안 통제 방안을 단계별로 설명하세요.

한국어 답변: """,

                "정보통신": """다음은 정보통신 관련 문제입니다. 정보통신기반 보호법의 요구사항을 정확히 적용하여 답변하세요.

**정보통신 답변 지침:**
- 정보통신기반 보호법의 구체적 요구사항과 조항 인용
- 보고 의무와 대응 절차의 명확한 단계별 설명
- 관련 기관과 업무 분장의 정확한 구분과 설명
- 집적된 정보통신시설의 보호 기준과 방법 포함

{few_shot_examples}

문제: {question}

정보통신기반 보호법에 따른 구체적 요구사항과 보고 절차를 단계별로 명확히 설명하세요.

한국어 답변: """
            }
        }

    def build_enhanced_prompt(self, question: str, question_type: str, domain: str = "일반", 
                             context_info: str = "", institution_info: str = "", force_diversity: bool = False) -> str:
        """향상된 프롬프트 구성 - 정확도 최적화"""
        try:
            # Few-shot 예시 개수 최적화
            example_count = self._determine_optimal_example_count(domain, question_type, question)
            few_shot_examples = self.build_few_shot_context(domain, question_type, question, count=example_count)
            
            # 특화된 질문 유형별 처리
            if self._is_ratio_question(question, domain):
                template = self.prompt_templates["ratio_specialized"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 기관 질문 특화 처리
            if self._is_institution_question(question, domain) and institution_info:
                template = self.prompt_templates["institution_specialized"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    institution_info=institution_info,
                    question=question
                )
            
            # 도메인 특화 템플릿 우선 사용
            if domain in self.prompt_templates["domain_specialized"] and question_type == "subjective":
                template = self.prompt_templates["domain_specialized"][domain]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 기본 템플릿 사용
            if question_type == "multiple_choice":
                template = self.prompt_templates["multiple_choice_enhanced"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            else:
                template = self.prompt_templates["subjective_enhanced"]
                enhanced_context = self._enhance_context_with_domain_expertise(context_info, domain, question)
                
                return template.format(
                    few_shot_examples=few_shot_examples,
                    context_info=enhanced_context,
                    question=question
                )
                
        except Exception as e:
            print(f"향상된 프롬프트 구성 오류: {e}")
            return self._create_robust_fallback_prompt(question, question_type, domain)

    def _determine_optimal_example_count(self, domain: str, question_type: str, question: str) -> int:
        """최적 예시 개수 결정 - 정확도 기반"""
        try:
            # 도메인별 기본 예시 개수 (정확도 최적화)
            domain_counts = {
                "사이버보안": 3,  # 복잡한 기술적 개념
                "전자금융": 2,   # 법령과 기관 정보
                "개인정보보호": 2, # 법적 절차
                "정보보안": 2,   # 관리체계 개념
                "위험관리": 1,   # 기본 개념
                "금융투자": 1,   # 분류 개념
                "정보통신": 1    # 보고 절차
            }
            
            base_count = domain_counts.get(domain, 1)
            
            # 질문 복잡도에 따른 조정
            complexity_indicators = ["특징", "지표", "방안", "절차", "기관", "비율"]
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in question.lower())
            
            if complexity_score >= 2:
                base_count = min(base_count + 1, 3)
            elif len(question) > 350:
                base_count = min(base_count + 1, 3)
            
            # 객관식의 경우 선택지 개수 고려
            if question_type == "multiple_choice":
                choice_count = len(re.findall(r'\n[1-5]\s', question))
                if choice_count >= 5:
                    base_count = max(base_count, 2)
            
            return min(base_count, 3)  # 최대 3개로 제한
        except Exception:
            return 2

    def _enhance_context_with_domain_expertise(self, context_info: str, domain: str, question: str) -> str:
        """도메인 전문 지식으로 컨텍스트 강화"""
        try:
            if not context_info:
                context_info = "관련 법령과 규정을 정확히 참고하여 답변하세요."
            
            # 도메인별 전문 컨텍스트 추가
            domain_expertise = {
                "사이버보안": {
                    "base": "사이버보안 위협 분석, 탐지 기술, 대응 방안을 기술적 관점에서 접근하세요.",
                    "keywords": {
                        "트로이": "트로이 목마의 위장 기법과 원격제어 메커니즘을 중심으로 설명하세요.",
                        "RAT": "원격접근도구의 동작 원리와 네트워크 통신 패턴을 포함하세요.",
                        "SBOM": "소프트웨어 구성 요소 관리와 공급망 보안 강화 관점을 강조하세요.",
                        "딥페이크": "AI 기반 위조 기술과 탐지 방법의 기술적 측면을 다루세요."
                    }
                },
                "전자금융": {
                    "base": "전자금융거래법과 전자금융감독규정의 구체적 조항을 정확히 적용하세요.",
                    "keywords": {
                        "분쟁조정": "전자금융거래법 제28조와 분쟁조정위원회의 권한을 명시하세요.",
                        "한국은행": "한국은행법 제91조의 자료제출 요구 권한을 정확히 인용하세요.",
                        "비율": "전자금융감독규정 제16조의 5%, 7% 기준을 명확히 제시하세요."
                    }
                },
                "개인정보보호": {
                    "base": "개인정보보호법의 처리 원칙과 정보주체 권리를 중심으로 설명하세요.",
                    "keywords": {
                        "만14세": "개인정보보호법 제22조 제6항의 법정대리인 동의를 명시하세요.",
                        "보호위원회": "개인정보보호법 제7조의 설치 근거와 권한을 설명하세요.",
                        "경영진": "개인정보 관리체계에서 최고경영진의 역할을 강조하세요."
                    }
                },
                "정보보안": {
                    "base": "정보보안관리체계의 체계적 접근과 보안 통제를 중심으로 설명하세요.",
                    "keywords": {
                        "3대요소": "기밀성, 무결성, 가용성의 정의와 상호관계를 명확히 하세요.",
                        "재해복구": "복구 절차, 비상연락체계, RTO 등 핵심 요소를 구분하세요."
                    }
                }
            }
            
            if domain in domain_expertise:
                expertise = domain_expertise[domain]
                enhanced_context = f"{context_info}\n\n전문 지침: {expertise['base']}"
                
                # 키워드별 세부 지침 추가
                question_lower = question.lower()
                for keyword, guidance in expertise["keywords"].items():
                    if keyword in question_lower:
                        enhanced_context += f"\n특별 지침: {guidance}"
                        break
                
                return enhanced_context[:800]
            
            return context_info
            
        except Exception:
            return context_info if context_info else "관련 법령과 규정을 정확히 적용하세요."

    def build_few_shot_context(self, domain: str, question_type: str, question: str, count: int = 2) -> str:
        """Few-shot 예시 구성 - 품질 최적화"""
        try:
            # 도메인에 해당하는 예시가 없으면 빈 문자열 반환
            if domain not in self.few_shot_examples:
                return ""
            
            domain_examples = self.few_shot_examples[domain]
            if question_type not in domain_examples:
                return ""
            
            examples = domain_examples[question_type]
            if not examples:
                return ""
            
            # 질문과의 유사도 기반 예시 선택
            selected_examples = self._select_most_relevant_examples(examples, question, count)
            
            # Few-shot 텍스트 생성
            few_shot_text = ""
            for i, example in enumerate(selected_examples, 1):
                if question_type == "multiple_choice":
                    few_shot_text += f"**참고 예시 {i}:**\n"
                    few_shot_text += f"문제: {example['question']}\n"
                    few_shot_text += f"정답: {example['answer']}\n"
                    few_shot_text += f"해설: {example['reasoning']}\n\n"
                else:
                    few_shot_text += f"**참고 예시 {i}:**\n"
                    few_shot_text += f"문제: {example['question']}\n"
                    few_shot_text += f"답변: {example['answer']}\n\n"
            
            return few_shot_text
            
        except Exception as e:
            print(f"Few-shot 컨텍스트 구성 오류: {e}")
            return ""

    def _select_most_relevant_examples(self, examples: List[Dict], question: str, count: int) -> List[Dict]:
        """가장 관련성 높은 예시 선택"""
        try:
            question_keywords = set(re.findall(r'[가-힣]+', question.lower()))
            
            scored_examples = []
            for example in examples:
                score = 0
                
                # 도메인 포커스 키워드 매칭
                if "domain_focus" in example:
                    focus_matches = sum(1 for focus in example["domain_focus"] 
                                      if focus.lower() in question.lower())
                    score += focus_matches * 3.0
                
                # 질문 키워드 유사성
                example_question = example.get("question", "")
                example_keywords = set(re.findall(r'[가-힣]+', example_question.lower()))
                
                if question_keywords and example_keywords:
                    intersection = question_keywords & example_keywords
                    union = question_keywords | example_keywords
                    jaccard_similarity = len(intersection) / len(union) if union else 0
                    score += jaccard_similarity * 2.0
                
                # 품질 점수 고려
                quality_score = example.get("quality_score", 0.8)
                score += quality_score * 1.5
                
                # 난이도 매칭
                if "difficulty" in example:
                    difficulty_bonus = 0.5
                    score += difficulty_bonus
                
                scored_examples.append((example, score))
            
            # 점수별 정렬 및 선택
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            selected = [ex for ex, _ in scored_examples[:count]]
            
            # 부족한 경우 추가 선택
            if len(selected) < count and len(examples) > len(selected):
                remaining = [ex for ex in examples if ex not in selected]
                selected.extend(remaining[:count - len(selected)])
            
            return selected[:count]
            
        except Exception:
            return examples[:count] if examples else []

    def _is_ratio_question(self, question: str, domain: str) -> bool:
        """비율 질문 확인 - 정확도 향상"""
        question_lower = question.lower()
        
        # 강력한 비율 지표
        strong_indicators = [
            r"비율.*얼마", r"기준.*몇.*%", r"정보기술부문.*비율", 
            r"예산.*비율", r"인력.*비율", r"\d+%.*이상", 
            r"배정.*기준", r"몇.*퍼센트"
        ]
        
        for indicator in strong_indicators:
            if re.search(indicator, question_lower):
                return True
        
        # 전자금융 도메인 특별 검사
        if domain == "전자금융":
            ratio_keywords = ["비율", "기준", "정보기술부문", "예산", "인력", "배정", "%", "퍼센트"]
            keyword_count = sum(1 for keyword in ratio_keywords if keyword in question_lower)
            if keyword_count >= 3:
                return True
        
        return False

    def _is_institution_question(self, question: str, domain: str) -> bool:
        """기관 질문 확인 - 정확도 향상"""
        question_lower = question.lower()
        
        # 강력한 기관 지표
        institution_indicators = [
            r"어떤.*기관", r"어느.*기관", r"기관.*기술하세요", r"기관.*설명하세요",
            r"분쟁조정.*신청.*기관", r"신고.*기관", r"상담.*기관", r"담당.*기관",
            r"소관.*기관", r"관할.*기관"
        ]
        
        for indicator in institution_indicators:
            if re.search(indicator, question_lower):
                return True
        
        # 도메인별 기관 키워드 검사
        domain_institution_keywords = {
            "전자금융": ["분쟁조정", "전자금융분쟁조정위원회", "금융감독원", "한국은행"],
            "개인정보보호": ["개인정보보호위원회", "침해신고센터", "신고", "상담"],
            "정보통신": ["과학기술정보통신부", "정보통신기반보호"]
        }
        
        if domain in domain_institution_keywords:
            keywords = domain_institution_keywords[domain]
            keyword_matches = sum(1 for keyword in keywords if keyword in question_lower)
            if keyword_matches >= 1 and "기관" in question_lower:
                return True
        
        return False

    def _create_robust_fallback_prompt(self, question: str, question_type: str, domain: str) -> str:
        """강화된 폴백 프롬프트"""
        try:
            if question_type == "multiple_choice":
                return f"""다음 금융보안 객관식 문제를 체계적으로 분석하여 정확한 답을 선택하세요.

**분석 단계:**
1. 질문 유형 파악: 긍정형 vs 부정형
2. 핵심 키워드 식별
3. 각 선택지의 타당성 검토
4. 관련 법령과 규정 적용
5. 논리적 추론을 통한 정답 도출

**중요 지침:**
- 부정형 질문("해당하지 않는", "적절하지 않은", "옳지 않은")의 경우 조건에 맞지 않는 선택지 찾기
- 긍정형 질문("가장 적절한", "올바른")의 경우 조건에 가장 부합하는 선택지 선택

문제: {question}

위 단계에 따라 체계적으로 분석한 후 정답 번호만 제시하세요.

정답 번호: """
            else:
                domain_guidance = {
                    "사이버보안": "기술적 메커니즘과 실무적 대응 방안을 중심으로",
                    "전자금융": "전자금융거래법의 구체적 조항과 절차를 중심으로",
                    "개인정보보호": "개인정보보호법의 원칙과 권리를 중심으로",
                    "정보보안": "정보보안관리체계와 보안 통제를 중심으로",
                    "정보통신": "정보통신기반 보호법의 요구사항을 중심으로"
                }.get(domain, "관련 법령과 규정을 중심으로")
                
                return f"""다음 금융보안 주관식 문제에 대해 정확하고 전문적인 한국어 답변을 작성하세요.

**필수 작성 지침:**
- 모든 답변은 한국어로만 작성 (영어 절대 금지)
- 관련 법령의 구체적 조항과 근거 명시
- {domain_guidance} 답변 구성
- 실무적이고 구체적인 내용 포함
- 정확한 전문용어 사용

문제: {question}

위 지침에 따라 관련 법령과 규정을 정확히 인용하면서 전문적인 한국어 답변을 작성하세요.

한국어 답변: """
                
        except Exception:
            return f"다음 문제에 정확하고 전문적으로 답변하세요:\n\n{question}\n\n답변: "

    def cleanup(self):
        """리소스 정리"""
        try:
            self.used_examples_cache.clear()
            self.prompt_history.clear()
        except Exception as e:
            print(f"프롬프트 enhancer 정리 오류: {e}")