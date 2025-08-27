# prompt_enhancer.py

import re
import random
import hashlib
from typing import Dict, List


class PromptEnhancer:
    """향상된 프롬프트 구성 및 Few-shot 예시 관리"""
    
    def __init__(self):
        self._initialize_enhanced_few_shot_examples()
        self._initialize_enhanced_prompt_templates()
        self._initialize_advanced_diversity_templates()
        self.used_examples_cache = {}
        self.prompt_history = []
        self.domain_performance = {}
        
    def _initialize_enhanced_few_shot_examples(self):
        """향상된 Few-shot 예시 초기화"""
        
        self.enhanced_few_shot_examples = {
            "사이버보안": {
                "multiple_choice": [
                    {
                        "question": "다음 중 트로이 목마의 주요 특징으로 가장 적절한 것은?\n1 자가 복제 기능\n2 정상 프로그램 위장\n3 네트워크 속도 저하\n4 파일 암호화\n5 화면 잠금",
                        "answer": "2",
                        "reasoning": "트로이 목마는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 것이 주요 특징입니다.",
                        "domain_focus": ["트로이", "특징", "위장"],
                        "difficulty": "중급"
                    },
                    {
                        "question": "SBOM을 금융권에서 활용하는 가장 적절한 이유는?\n1 데이터 백업\n2 네트워크 모니터링\n3 접근 권한 관리\n4 암호화 강화\n5 소프트웨어 공급망 보안",
                        "answer": "5",
                        "reasoning": "SBOM은 소프트웨어 구성 요소의 투명성을 제공하여 공급망 보안을 강화하는 목적으로 활용됩니다.",
                        "domain_focus": ["SBOM", "활용", "공급망"],
                        "difficulty": "고급"
                    },
                    {
                        "question": "딥페이크 기술의 악용을 방지하기 위한 금융권의 선제적 대응 방안으로 가장 적절한 것은?\n1 딥페이크 탐지 기능이 없는 구식 인증 시스템 도입\n2 딥보이스 탐지 기술 개발\n3 금융기관의 음성 복제\n4 딥페이크 영상 제작 지원\n5 금융소비자 홍보 강화",
                        "answer": "2",
                        "reasoning": "딥페이크 기술 악용 방지를 위한 선제적 대응 방안으로는 딥보이스 탐지 기술 개발이 가장 적절합니다.",
                        "domain_focus": ["딥페이크", "대응", "탐지"],
                        "difficulty": "고급"
                    },
                    {
                        "question": "디지털 지갑에서 우려되는 주요 보안 위협으로 가장 적절하지 않은 것은?\n1 개인키 도난\n2 피싱 공격\n3 멀웨어 감염\n4 스마트 컨트랙트 취약점\n5 네트워크 속도 저하",
                        "answer": "5",
                        "reasoning": "네트워크 속도 저하는 디지털 지갑의 주요 보안 위협이 아니며, 다른 선택지들은 모두 실질적인 보안 위협입니다.",
                        "domain_focus": ["디지털지갑", "보안위협"],
                        "difficulty": "중급"
                    }
                ],
                "subjective": [
                    {
                        "question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요.",
                        "answer": "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하고 외부 공격자가 원격으로 시스템을 제어할 수 있도록 하는 특성을 가집니다. 주요 탐지 지표로는 비정상적인 네트워크 통신 패턴, 비인가 프로세스 실행, 파일 시스템 변경, 레지스트리 수정, 시스템 리소스 과다 사용 등이 있으며, 실시간 모니터링과 행동 분석을 통한 종합적 탐지 및 즉시 차단이 필요합니다.",
                        "domain_focus": ["트로이", "특징", "탐지지표"],
                        "answer_type": "특징설명",
                        "length_category": "긴",
                        "quality_score": 0.9
                    },
                    {
                        "question": "딥페이크 기술 악용에 대비한 금융권의 대응 방안을 기술하세요.",
                        "answer": "딥페이크 기술 악용에 대비하여 금융권에서는 다층 방어체계 구축, 딥보이스 탐지 기술 개발 및 도입, 생체인증과 다중 인증 체계를 통한 신원 검증 강화, 직원 교육 및 고객 인식 제고, 실시간 모니터링 시스템 구축을 통한 선제적 보안 대응 방안을 수립하고 지속적으로 개선해야 합니다.",
                        "domain_focus": ["딥페이크", "대응", "방안"],
                        "answer_type": "방안제시",
                        "length_category": "중간",
                        "quality_score": 0.85
                    },
                    {
                        "question": "디지털 지갑(Digital Wallet)에서 우려되는 주요 보안 위협을 설명하세요.",
                        "answer": "디지털 지갑의 주요 보안 위협으로는 개인키 도난 및 분실, 피싱 및 스미싱 공격, 멀웨어 감염, 스마트 컨트랙트 취약점, 거래소 해킹, 중간자 공격 등이 있으며, 이에 대응하기 위해 다중 인증 시스템 도입, 하드웨어 지갑 사용, 정기적인 보안 업데이트, 안전한 네트워크 환경 사용이 권장됩니다.",
                        "domain_focus": ["디지털지갑", "보안위협"],
                        "answer_type": "위협설명",
                        "length_category": "중간",
                        "quality_score": 0.8
                    }
                ]
            },
            "전자금융": {
                "multiple_choice": [
                    {
                        "question": "한국은행이 금융통화위원회의 요청에 따라 전자금융업자에게 자료제출을 요구할 수 있는 경우로 가장 적절한 것은?\n1 보안 시스템 점검\n2 고객 정보 확인\n3 경영 실적 조사\n4 통화신용정책 수행\n5 시장 동향 파악",
                        "answer": "4",
                        "reasoning": "한국은행법에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있습니다.",
                        "domain_focus": ["한국은행", "자료제출", "통화신용정책"],
                        "difficulty": "고급"
                    },
                    {
                        "question": "전자금융감독규정에 따른 금융회사의 정보보호 예산 배정 비율로 가장 적절한 것은?\n1 정보기술부문 인력 3% 이상, 예산 5% 이상\n2 정보기술부문 인력 5% 이상, 예산 7% 이상\n3 정보기술부문 인력 7% 이상, 예산 5% 이상\n4 정보기술부문 인력 10% 이상, 예산 10% 이상\n5 별도의 기준 없음",
                        "answer": "2",
                        "reasoning": "전자금융감독규정 제16조에 따라 정보기술부문 인력 5% 이상, 예산 7% 이상을 정보보호 업무에 배정해야 합니다.",
                        "domain_focus": ["정보기술부문", "비율", "예산"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요.",
                        "answer": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당하며, 금융감독원 내에 설치되어 전자금융거래법 제28조에 근거하여 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하는 역할을 수행합니다. 조정 신청은 온라인(www.fss.or.kr) 또는 서면으로 가능하며, 조정 절차는 무료로 진행됩니다.",
                        "domain_focus": ["분쟁조정", "기관", "전자금융거래법"],
                        "answer_type": "기관명",
                        "length_category": "중간",
                        "quality_score": 0.9
                    },
                    {
                        "question": "금융회사가 정보보호 예산을 관리할 때, 전자금융감독규정상 정보기술부문 인력 및 예산의 기준 비율은 얼마인가요?",
                        "answer": "전자금융감독규정 제16조에 따라 금융회사는 정보기술부문 인력을 총 인력의 5% 이상, 정보기술부문 예산을 총 예산의 7% 이상 정보보호 업무에 배정해야 합니다. 다만 회사 규모, 업무 특성, 정보기술 위험수준 등을 고려하여 금융감독원장이 별도로 정할 수 있으며, 이는 금융회사의 정보보호 역량 강화를 위한 최소 기준입니다.",
                        "domain_focus": ["정보기술부문", "비율", "예산"],
                        "answer_type": "수치설명",
                        "length_category": "긴",
                        "quality_score": 0.95
                    },
                    {
                        "question": "전자금융업자가 수행해야 할 보안조치의 주요 내용을 설명하세요.",
                        "answer": "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 포괄적인 보안조치를 시행해야 합니다. 주요 내용으로는 접근매체의 안전한 보관 및 관리, 거래기록의 보존과 위조변조 방지, 암호화 기술을 통한 거래정보 보호, 이용자 인증 강화, 보안프로그램 설치 및 운영 등 종합적인 보안체계를 구축하고 지속적으로 관리해야 합니다.",
                        "domain_focus": ["보안조치", "전자금융업자"],
                        "answer_type": "조치설명",
                        "length_category": "중간",
                        "quality_score": 0.85
                    }
                ]
            },
            "개인정보보호": {
                "multiple_choice": [
                    {
                        "question": "만 14세 미만 아동의 개인정보 처리를 위해 필요한 절차로 가장 적절한 것은?\n1 본인의 직접 동의\n2 법정대리인의 동의\n3 학교의 승인\n4 관할 기관 허가\n5 보호자 확인서",
                        "answer": "2",
                        "reasoning": "개인정보보호법 제22조에 따라 만 14세 미만 아동의 개인정보 처리에는 법정대리인의 동의가 필요합니다.",
                        "domain_focus": ["만14세", "아동", "법정대리인"],
                        "difficulty": "중급"
                    },
                    {
                        "question": "'관리체계 수립 및 운영'의 '정책 수립' 단계에서 가장 중요한 요소는 무엇인가?\n1 정보보호 및 개인정보보호 정책의 제·개정\n2 경영진의 참여\n3 최고책임자의 지정\n4 자원 할당\n5 내부 감사 절차의 수립",
                        "answer": "2",
                        "reasoning": "정책 수립 단계에서는 경영진의 적극적인 참여와 의지가 가장 중요한 요소입니다.",
                        "domain_focus": ["정책수립", "경영진", "중요한요소"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "개인정보 접근 권한 검토는 어떻게 수행해야 하며, 그 목적은 무엇인가요?",
                        "answer": "개인정보 접근 권한 검토는 업무상 필요한 최소한의 권한만을 부여하는 최소권한 원칙에 따라 정기적으로 수행해야 하며, 불필요한 권한은 즉시 회수하고 접근 로그를 지속적으로 관리하여 개인정보 오남용을 방지하고 정보보안을 강화하는 것이 목적입니다. 또한 권한 변경 시 승인 절차를 거쳐야 하며, 퇴직자나 부서 이동자의 권한은 즉시 회수해야 합니다.",
                        "domain_focus": ["접근권한", "검토", "최소권한"],
                        "answer_type": "절차설명",
                        "length_category": "중간",
                        "quality_score": 0.85
                    },
                    {
                        "question": "개인정보 관리체계 수립 및 운영의 정책 수립 단계에서 가장 중요한 요소를 설명하세요.",
                        "answer": "개인정보 관리체계의 정책 수립 단계에서 가장 중요한 요소는 경영진의 적극적인 참여와 의지입니다. 최고 경영진의 개인정보보호에 대한 확고한 의지와 충분한 자원 지원이 있어야 체계적이고 효과적인 관리체계를 구축할 수 있으며, 조직 전체의 개인정보보호 문화를 정착시킬 수 있습니다. 경영진의 리더십 없이는 실효성 있는 정책 수립과 지속적인 운영이 어렵습니다.",
                        "domain_focus": ["관리체계", "정책수립", "경영진"],
                        "answer_type": "요소설명",
                        "length_category": "중간",
                        "quality_score": 0.9
                    },
                    {
                        "question": "개인정보보호위원회의 주요 업무와 개인정보침해신고센터의 역할을 기술하세요.",
                        "answer": "개인정보보호위원회는 개인정보보호법에 따라 개인정보 보호에 관한 업무를 총괄하는 국가기관으로서 개인정보보호 정책 수립, 법령 제·개정, 개인정보 처리 실태 점검 및 감독 업무를 수행합니다. 개인정보침해신고센터(privacy.go.kr)는 개인정보 침해신고 접수 및 상담 업무를 담당하며, 개인정보 처리와 관련된 분쟁조정 및 집단분쟁조정도 수행하여 개인정보보호 업무의 실질적 창구 역할을 합니다.",
                        "domain_focus": ["보호위원회", "신고센터", "업무"],
                        "answer_type": "기관설명",
                        "length_category": "긴",
                        "quality_score": 0.9
                    }
                ]
            },
            "정보보안": {
                "multiple_choice": [
                    {
                        "question": "재해 복구 계획 수립 시 고려 요소 중 옳지 않은 것은?\n1 복구 절차 수립\n2 비상연락체계 구축\n3 개인정보 파기 절차\n4 복구 목표시간 설정\n5 백업 시스템 구축",
                        "answer": "3",
                        "reasoning": "개인정보 파기 절차는 재해 복구 계획과 직접적인 관련이 없으며, 복구 관련 요소가 아닙니다.",
                        "domain_focus": ["재해복구", "계획수립", "파기절차"],
                        "difficulty": "중급"
                    },
                    {
                        "question": "정보보호의 3대 요소가 아닌 것은?\n1 기밀성(Confidentiality)\n2 무결성(Integrity)\n3 가용성(Availability)\n4 확장성(Scalability)\n5 모두 3대 요소에 해당",
                        "answer": "4",
                        "reasoning": "정보보호의 3대 요소는 기밀성, 무결성, 가용성이며, 확장성은 해당되지 않습니다.",
                        "domain_focus": ["3대요소", "정보보호"],
                        "difficulty": "초급"
                    }
                ],
                "subjective": [
                    {
                        "question": "정보보호의 3대 요소에 해당하는 보안 목표를 3가지 기술하세요.",
                        "answer": "정보보호의 3대 요소는 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)으로 구성됩니다. 기밀성은 인가된 사용자만이 정보에 접근할 수 있도록 하는 것이며, 무결성은 정보의 정확성과 완전성을 보장하여 비인가된 변경을 방지하는 것입니다. 가용성은 인가된 사용자가 필요할 때 언제든지 정보와 자원에 접근할 수 있도록 보장하는 것으로, 이 세 요소를 통해 종합적인 정보보안 체계를 구축할 수 있습니다.",
                        "domain_focus": ["3대요소", "보안목표"],
                        "answer_type": "정의설명",
                        "length_category": "중간",
                        "quality_score": 0.9
                    },
                    {
                        "question": "SMTP 프로토콜의 보안상 주요 역할을 설명하세요.",
                        "answer": "SMTP(Simple Mail Transfer Protocol) 프로토콜은 이메일 전송을 담당하며, 보안상 주요 역할로는 SMTP AUTH를 통한 사용자 인증 메커니즘 제공, STARTTLS를 통한 암호화 통신 지원, 스팸 및 악성 이메일 차단 기능을 통해 안전하고 신뢰할 수 있는 이메일 서비스를 보장합니다. 또한 발신자 검증과 메일 서버 간 보안 연결을 통해 이메일 통신의 기밀성과 무결성을 확보하는 역할을 수행합니다.",
                        "domain_focus": ["SMTP", "프로토콜", "보안역할"],
                        "answer_type": "역할설명",
                        "length_category": "중간",
                        "quality_score": 0.85
                    }
                ]
            },
            "정보통신": {
                "multiple_choice": [
                    {
                        "question": "집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 과학기술정보통신부장관에게 보고해야 하는 사항으로 옳지 않은 것은?\n1 정보통신서비스 제공의 중단이 발생한 일시 및 장소\n2 정보통신서비스 제공의 중단이 발생한 원인에 대한 법적 책임\n3 정보통신서비스 제공의 중단이 발생한 원인 및 피해내용\n4 응급조치 사항",
                        "answer": "2",
                        "reasoning": "정보통신서비스 제공 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항에는 법적 책임이 포함되지 않습니다.",
                        "domain_focus": ["정보통신서비스", "중단", "보고사항"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "정보통신시설의 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항을 설명하세요.",
                        "answer": "집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 정보통신기반 보호법에 따라 과학기술정보통신부장관에게 보고해야 하는 사항은 중단이 발생한 일시 및 장소, 중단이 발생한 원인 및 피해내용, 응급조치 사항입니다. 다만 법적 책임에 관한 사항은 보고 대상에 해당하지 않으며, 신속하고 정확한 현황 보고를 통해 효과적인 대응 방안을 마련하는 것이 중요합니다.",
                        "domain_focus": ["정보통신시설", "중단보고", "보고사항"],
                        "answer_type": "절차설명",
                        "length_category": "중간",
                        "quality_score": 0.9
                    }
                ]
            },
            "금융투자": {
                "multiple_choice": [
                    {
                        "question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업",
                        "answer": "1",
                        "reasoning": "소비자금융업은 금융투자업에 해당하지 않으며, 별도의 금융업 분류에 속합니다.",
                        "domain_focus": ["금융투자업", "구분", "소비자금융업"],
                        "difficulty": "중급"
                    }
                ]
            },
            "위험관리": {
                "multiple_choice": [
                    {
                        "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간",
                        "answer": "2",
                        "reasoning": "위험 관리 계획에서 위험 수용은 적절한 관리 요소가 아니며, 위험을 식별하고 대응하는 것이 중요합니다.",
                        "domain_focus": ["위험관리", "계획수립", "위험수용"],
                        "difficulty": "중급"
                    }
                ]
            }
        }
    
    def _initialize_enhanced_prompt_templates(self):
        """향상된 프롬프트 템플릿 초기화"""
        
        self.enhanced_prompt_templates = {
            "multiple_choice_advanced": """다음은 금융보안 관련 객관식 문제입니다. 체계적 분석을 통해 가장 적절한 답을 선택하세요.

{few_shot_examples}

문제 분석 단계:
1. 질문의 핵심 키워드와 요구사항 정확히 파악
2. 각 선택지를 해당 법령과 규정에 따라 면밀히 검토
3. 문제 유형(부정/긍정)에 따른 논리적 추론 적용
4. 도메인 전문 지식을 바탕으로 최적 답안 선택

문제: {question}

위 단계를 따라 체계적으로 분석하여 정답 번호만 제시하세요.

정답 번호: """,

            "subjective_advanced": """다음은 금융보안 관련 주관식 문제입니다. 반드시 한국어로만 전문적이고 정확한 답변을 작성하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성 (영어 사용 절대 금지)
- 관련 법령과 규정에 근거한 전문적 답변 작성
- 구체적이고 실무적인 내용 포함
- 자연스러운 한국어 문장으로 구성
- 도메인별 전문용어 적절히 활용{diversity_instruction}

{few_shot_examples}

참고 정보:
{context_info}

문제: {question}

위 문제에 대해 관련 법령과 규정을 근거로 구체적이고 전문적인 한국어 답변을 작성하세요.

한국어 답변: """,

            "institution_specialized": """다음은 금융보안 관련 기관에 대한 질문입니다. 정확한 기관 정보를 바탕으로 한국어로만 답변하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성
- 기관의 정확한 명칭과 역할 기술
- 법적 근거와 설립 배경 포함
- 구체적인 업무 범위와 절차 설명

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

            "ratio_specialized": """다음은 금융보안 관련 비율에 대한 질문입니다. 구체적인 수치와 법적 근거를 포함하여 한국어로만 답변하세요.

답변 작성 지침:
- 정확한 수치와 퍼센트 명시
- 해당 법령과 조항 인용
- 예외 조건이나 특별 규정 포함
- 적용 범위와 기준 명확히 설명

{few_shot_examples}

문제: {question}

위 질문에 대해 다음 사항을 포함하여 한국어로 답변하세요:
1. 정확한 비율과 수치
2. 관련 법령과 조항 번호
3. 적용 조건 및 예외사항
4. 감독기관의 재량권과 기준

한국어 답변: """,

            "domain_specialized": {
                "사이버보안": """다음은 사이버보안 관련 문제입니다. 기술적 특성과 보안 대응 방안에 중점을 두어 답변하세요.

답변 작성 지침:
- 기술적 메커니즘과 동작 원리 상세 설명
- 실제 위협 사례와 탐지 방법 포함
- 다층 방어체계와 종합적 대응방안 제시

{few_shot_examples}

문제: {question}

사이버보안 전문가 관점에서 기술적 특징, 탐지 방법, 대응 방안을 구체적으로 설명하세요.

한국어 답변: """,

                "전자금융": """다음은 전자금융 관련 문제입니다. 전자금융거래법과 관련 규정을 근거로 답변하세요.

답변 작성 지침:
- 전자금융거래법의 구체적 조항 인용
- 이용자 보호와 업무 절차 중심 설명
- 법적 요구사항과 실무 적용 방법 제시

{few_shot_examples}

문제: {question}

전자금융거래법과 관련 규정에 근거하여 법적 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """,

                "개인정보보호": """다음은 개인정보보호 관련 문제입니다. 개인정보보호법을 근거로 답변하세요.

답변 작성 지침:
- 개인정보보호법의 원칙과 절차 중심 설명
- 정보주체의 권리와 처리자의 의무 명시
- 실무 적용 사례와 주의사항 포함

{few_shot_examples}

문제: {question}

개인정보보호법에 따른 처리 원칙과 절차를 구체적으로 설명하세요.

한국어 답변: """,

                "정보보안": """다음은 정보보안 관련 문제입니다. 정보보안관리체계 관점에서 답변하세요.

답변 작성 지침:
- 정보보안관리체계의 체계적 접근 중심 설명
- 기술적·관리적·물리적 보안대책 구분 제시
- 위험 분석과 단계별 대응 방안 포함

{few_shot_examples}

문제: {question}

정보보안관리체계 구축과 운영 관점에서 체계적으로 설명하세요.

한국어 답변: """,

                "정보통신": """다음은 정보통신 관련 문제입니다. 정보통신기반 보호법을 근거로 답변하세요.

답변 작성 지침:
- 정보통신기반 보호법의 요구사항 중심 설명
- 보고 의무와 대응 절차 명확히 제시
- 관련 기관과 업무 분장 포함

{few_shot_examples}

문제: {question}

정보통신기반 보호법에 따른 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """
            }
        }

    def _initialize_advanced_diversity_templates(self):
        """고급 다양성 확보 템플릿 초기화"""
        
        self.advanced_diversity_instructions = {
            "legal_comprehensive": "\n- 법적 조항과 규정을 종합적으로 분석하여 답변하세요.",
            "practical_implementation": "\n- 실무 적용과 구체적인 구현 방법을 중심으로 답변하세요.", 
            "technical_deep_dive": "\n- 기술적 특성과 심층적인 메커니즘을 상세히 설명하세요.",
            "process_systematic": "\n- 체계적인 단계별 과정과 절차를 명확히 제시하세요.",
            "comprehensive_analysis": "\n- 다각적이고 종합적인 분석을 통해 포괄적으로 답변하세요.",
            "risk_assessment": "\n- 위험 요소 분석과 예방 방안을 중심으로 답변하세요.",
            "stakeholder_perspective": "\n- 이해관계자별 관점과 역할을 고려하여 답변하세요."
        }
        
        self.contextual_variations = {
            "authoritative": "법령과 규정에 근거한 공식적이고 권위있는 관점에서",
            "practical_expert": "실무 적용과 현장 전문성을 바탕으로",
            "systematic_analyst": "체계적이고 논리적인 분석을 통해", 
            "comprehensive_advisor": "다각적이고 종합적인 자문 관점에서",
            "implementation_focused": "구체적인 실행과 적용 방법을 중심으로",
            "risk_aware": "위험 인식과 예방 중심의 관점에서",
            "compliance_oriented": "준수 사항과 규정 이행을 중시하는 관점에서"
        }

        # 도메인별 전문가 관점
        self.domain_expert_perspectives = {
            "사이버보안": [
                "사이버보안 전문가 관점에서 위협 분석과 대응 전략을",
                "정보보안 아키텍트 관점에서 시스템 설계와 보안 체계를",
                "보안 운영 전문가 관점에서 실시간 모니터링과 대응 절차를"
            ],
            "전자금융": [
                "전자금융 규제 전문가 관점에서 법적 요구사항과 준수 방안을",
                "핀테크 보안 전문가 관점에서 기술적 보안 조치와 위험 관리를",
                "금융감독 실무자 관점에서 검사 기준과 평가 요소를"
            ],
            "개인정보보호": [
                "개인정보보호 전문가 관점에서 법적 의무와 권리 보장 방안을",
                "프라이버시 엔지니어 관점에서 기술적 보호 조치와 시스템 설계를",
                "개인정보보호 감사관 관점에서 점검 기준과 평가 방법을"
            ],
            "정보보안": [
                "정보보안 관리자 관점에서 체계적인 보안 정책과 절차를",
                "보안 컨설턴트 관점에서 위험 분석과 개선 방안을",
                "ISMS 심사원 관점에서 관리체계 구축과 운영 요구사항을"
            ]
        }

    def _generate_enhanced_prompt_hash(self, question: str, domain: str, question_type: str) -> str:
        """향상된 프롬프트 해시 생성"""
        try:
            combined_text = f"{question[:100]}-{domain}-{question_type}-{len(question)}"
            return hashlib.md5(combined_text.encode()).hexdigest()[:10]
        except Exception:
            return ""

    def _select_high_quality_examples(self, domain: str, question_type: str, question: str, count: int = 2) -> List[Dict]:
        """고품질 예시 선택"""
        try:
            if domain not in self.enhanced_few_shot_examples:
                return []
            
            domain_examples = self.enhanced_few_shot_examples[domain]
            if question_type not in domain_examples:
                return []
            
            examples = domain_examples[question_type]
            if not examples:
                return []
            
            question_keywords = set(question.lower().split())
            question_difficulty = self._assess_question_difficulty(question)
            
            # 예시별 점수 계산
            scored_examples = []
            for example in examples:
                score = 0
                
                # 도메인 포커스 키워드 매칭
                if "domain_focus" in example:
                    focus_matches = sum(1 for focus in example["domain_focus"] 
                                      if focus.lower() in question.lower())
                    score += focus_matches * 3.0
                
                # 난이도 매칭
                if "difficulty" in example:
                    if example["difficulty"] == question_difficulty:
                        score += 5.0
                    elif abs(self._difficulty_to_score(example["difficulty"]) - 
                           self._difficulty_to_score(question_difficulty)) <= 1:
                        score += 2.0
                
                # 답변 유형 매칭 (주관식)
                if question_type == "subjective":
                    if "answer_type" in example:
                        if self._match_answer_type_enhanced(question, example["answer_type"]):
                            score += 4.0
                    
                    # 품질 점수
                    if "quality_score" in example:
                        score += example["quality_score"] * 3.0
                
                # 길이 다양성 고려
                if "length_category" in example:
                    expected_length = self._estimate_expected_length(question)
                    if example["length_category"] == expected_length:
                        score += 2.0
                
                scored_examples.append((example, score))
            
            # 점수별 정렬
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            
            # 다양성 확보하여 선택
            selected_examples = []
            used_types = set()
            used_lengths = set()
            
            for example, score in scored_examples:
                if len(selected_examples) >= count:
                    break
                
                # 다양성 체크
                answer_type = example.get("answer_type", "default")
                length_cat = example.get("length_category", "default")
                
                if question_type == "subjective":
                    if answer_type in used_types and len(selected_examples) > 0:
                        continue
                    if length_cat in used_lengths and len(selected_examples) > 1:
                        continue
                
                selected_examples.append(example)
                used_types.add(answer_type)
                used_lengths.add(length_cat)
            
            # 부족한 경우 추가 선택
            if len(selected_examples) < count:
                remaining = [ex for ex, _ in scored_examples if ex not in selected_examples]
                selected_examples.extend(remaining[:count - len(selected_examples)])
            
            return selected_examples[:count]
            
        except Exception as e:
            print(f"고품질 예시 선택 오류: {e}")
            return examples[:count] if examples else []

    def _assess_question_difficulty(self, question: str) -> str:
        """질문 난이도 평가"""
        try:
            question_lower = question.lower()
            difficulty_score = 0
            
            # 기술 용어
            tech_terms = ["isms", "sbom", "rat", "딥페이크", "전자금융감독규정", "개인정보영향평가"]
            difficulty_score += sum(2 for term in tech_terms if term in question_lower)
            
            # 법령 관련
            legal_terms = ["법", "조", "항", "규정"]  
            difficulty_score += sum(1 for term in legal_terms if term in question_lower)
            
            # 복합성
            if "특징" in question_lower and "지표" in question_lower:
                difficulty_score += 3
            
            # 길이
            if len(question) > 300:
                difficulty_score += 2
            elif len(question) > 200:
                difficulty_score += 1
            
            if difficulty_score >= 6:
                return "고급"
            elif difficulty_score >= 3:
                return "중급"
            else:
                return "초급"
                
        except Exception:
            return "중급"

    def _difficulty_to_score(self, difficulty: str) -> int:
        """난이도를 점수로 변환"""
        mapping = {"초급": 1, "중급": 2, "고급": 3}
        return mapping.get(difficulty, 2)

    def _estimate_expected_length(self, question: str) -> str:
        """예상 답변 길이 추정"""
        try:
            question_lower = question.lower()
            
            if any(word in question_lower for word in ["비율", "얼마", "기관", "무엇"]):
                return "짧음"
            elif any(word in question_lower for word in ["특징", "지표", "방안", "절차"]):
                return "중간"
            elif any(word in question_lower for word in ["설명하세요", "기술하세요", "대응.*방안"]):
                return "긴"
            else:
                return "중간"
        except Exception:
            return "중간"

    def _match_answer_type_enhanced(self, question: str, answer_type: str) -> bool:
        """향상된 답변 유형 매칭"""
        question_lower = question.lower()
        
        enhanced_type_patterns = {
            "기관명": ["기관", "위원회", "담당", "어디", "누구", "신청"],
            "특징설명": ["특징", "특성", "성질", "어떤", "주요"],
            "지표나열": ["지표", "징후", "탐지", "모니터링", "패턴"],
            "방안제시": ["방안", "대책", "대응", "해결", "어떻게"],
            "절차설명": ["절차", "과정", "단계", "순서", "프로세스"],
            "수치설명": ["비율", "얼마", "기준", "퍼센트", "%"],
            "정의설명": ["정의", "무엇", "개념", "의미", "뜻"],
            "역할설명": ["역할", "기능", "업무", "담당", "수행"],
            "요소설명": ["요소", "구성", "핵심", "주요", "중요한"],
            "위협설명": ["위협", "위험", "우려", "취약점"],
            "복합설명": ["특징.*지표", "방안.*절차", "역할.*기능"]
        }
        
        if answer_type in enhanced_type_patterns:
            patterns = enhanced_type_patterns[answer_type]
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return True
        
        return False

    def build_enhanced_few_shot_context(self, domain: str, question_type: str, question: str, count: int = 2) -> str:
        """향상된 Few-shot 예시 구성"""
        try:
            # 프롬프트 해시 생성
            prompt_hash = self._generate_enhanced_prompt_hash(question, domain, question_type)
            
            # 캐시된 예시 확인 및 제외
            excluded_examples = self.used_examples_cache.get(prompt_hash, [])
            
            if question_type == "subjective":
                count = min(count, 2)  # 주관식은 최대 2개
            
            selected_examples = self._select_high_quality_examples(domain, question_type, question, count)
            
            # 제외 목록 확인
            if excluded_examples:
                available_examples = []
                for ex in selected_examples:
                    answer_signature = self._generate_example_signature(ex)
                    if answer_signature not in excluded_examples:
                        available_examples.append(ex)
                
                if available_examples:
                    selected_examples = available_examples
            
            # 사용된 예시 캐시 업데이트
            if selected_examples:
                used_signatures = [self._generate_example_signature(ex) for ex in selected_examples]
                self.used_examples_cache[prompt_hash] = used_signatures
                
                # 캐시 크기 제한
                if len(self.used_examples_cache) > 100:
                    oldest_key = list(self.used_examples_cache.keys())[0]
                    del self.used_examples_cache[oldest_key]
            
            # Few-shot 텍스트 생성
            few_shot_text = ""
            for i, example in enumerate(selected_examples, 1):
                if question_type == "multiple_choice":
                    few_shot_text += f"예시 {i}:\n문제: {example['question']}\n정답: {example['answer']}\n해설: {example['reasoning']}\n\n"
                else:
                    few_shot_text += f"예시 {i}:\n문제: {example['question']}\n답변: {example['answer']}\n\n"
            
            return few_shot_text
            
        except Exception as e:
            print(f"향상된 Few-shot 컨텍스트 구성 오류: {e}")
            return ""

    def _generate_example_signature(self, example: Dict) -> str:
        """예시 서명 생성"""
        try:
            answer = example.get("answer", "")
            question = example.get("question", "")
            return hashlib.md5(f"{question[:50]}{answer[:50]}".encode()).hexdigest()[:8]
        except Exception:
            return ""

    def _get_advanced_diversity_instruction(self, domain: str, force_diversity: bool = False) -> str:
        """고급 다양성 지침 선택"""
        if not force_diversity:
            return ""
        
        try:
            # 도메인별 선호 지침
            domain_preferences = {
                "사이버보안": ["technical_deep_dive", "risk_assessment", "comprehensive_analysis"],
                "전자금융": ["legal_comprehensive", "practical_implementation", "compliance_oriented"],
                "개인정보보호": ["legal_comprehensive", "stakeholder_perspective", "process_systematic"],
                "정보보안": ["systematic_analyst", "risk_assessment", "implementation_focused"],
                "위험관리": ["comprehensive_analysis", "risk_assessment", "process_systematic"],
                "금융투자": ["legal_comprehensive", "practical_implementation"],
                "정보통신": ["legal_comprehensive", "process_systematic"]
            }
            
            available_instructions = domain_preferences.get(
                domain, list(self.advanced_diversity_instructions.keys())
            )
            selected_instruction = random.choice(available_instructions)
            
            return self.advanced_diversity_instructions.get(selected_instruction, "")
            
        except Exception:
            return ""

    def _add_expert_perspective(self, base_prompt: str, domain: str, question_type: str) -> str:
        """전문가 관점 추가"""
        try:
            if domain in self.domain_expert_perspectives and question_type == "subjective":
                perspectives = self.domain_expert_perspectives[domain]
                selected_perspective = random.choice(perspectives)
                
                # 프롬프트에 전문가 관점 추가
                enhanced_prompt = base_prompt.replace(
                    "위 문제에 대해", 
                    f"{selected_perspective} 중심으로 위 문제에 대해"
                )
                return enhanced_prompt
            
            return base_prompt
        except Exception:
            return base_prompt

    def build_enhanced_prompt(self, question: str, question_type: str, domain: str = "일반", 
                            context_info: str = "", institution_info: str = "", 
                            force_diversity: bool = False) -> str:
        """향상된 프롬프트 구성"""
        try:
            # Few-shot 예시 추가 (품질 기반)
            example_count = self._determine_example_count(domain, question_type, question)
            few_shot_examples = self.build_enhanced_few_shot_context(domain, question_type, question, count=example_count)
            
            # 고급 다양성 지침 생성
            diversity_instruction = self._get_advanced_diversity_instruction(domain, force_diversity)
            
            # 특화된 질문 유형 처리
            if self._is_ratio_question_enhanced(question, domain):
                template = self.enhanced_prompt_templates["ratio_specialized"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 기관 질문 특화 처리
            if self._is_institution_question_enhanced(question, domain) and institution_info:
                template = self.enhanced_prompt_templates["institution_specialized"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    institution_info=institution_info,
                    question=question
                )
            
            # 도메인 특화 템플릿 사용
            if domain in self.enhanced_prompt_templates["domain_specialized"] and question_type == "subjective":
                template = self.enhanced_prompt_templates["domain_specialized"][domain]
                enhanced_template = self._add_expert_perspective(template, domain, question_type)
                return enhanced_template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 일반 향상된 프롬프트
            if question_type == "multiple_choice":
                template = self.enhanced_prompt_templates["multiple_choice_advanced"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            else:
                template = self.enhanced_prompt_templates["subjective_advanced"]
                enhanced_context = self._enhance_context_info(context_info, domain)
                
                enhanced_template = self._add_expert_perspective(template, domain, question_type)
                
                return enhanced_template.format(
                    few_shot_examples=few_shot_examples,
                    context_info=enhanced_context,
                    question=question,
                    diversity_instruction=diversity_instruction
                )
                
        except Exception as e:
            print(f"향상된 프롬프트 구성 오류: {e}")
            return self._create_fallback_enhanced_prompt(question, question_type, domain, force_diversity)

    def _determine_example_count(self, domain: str, question_type: str, question: str) -> int:
        """예시 개수 결정"""
        try:
            # 도메인별 기본 예시 개수
            domain_counts = {
                "사이버보안": 2,
                "전자금융": 2,
                "개인정보보호": 2,
                "정보보안": 1,
                "위험관리": 1,
                "금융투자": 1,
                "정보통신": 1
            }
            
            base_count = domain_counts.get(domain, 1)
            
            # 질문 복잡도에 따른 조정
            if len(question) > 300:
                base_count = min(base_count + 1, 3)
            elif any(word in question.lower() for word in ["특징", "지표", "방안", "절차"]):
                base_count = min(base_count + 1, 2)
            
            return base_count
        except Exception:
            return 1

    def _enhance_context_info(self, context_info: str, domain: str) -> str:
        """컨텍스트 정보 향상"""
        try:
            if not context_info:
                context_info = "관련 법령과 규정을 참고하세요."
            
            # 도메인별 추가 컨텍스트
            domain_contexts = {
                "사이버보안": "사이버보안 위협 분석 및 대응 기술 관점에서 접근하세요.",
                "전자금융": "전자금융거래법과 전자금융감독규정의 요구사항을 중심으로 고려하세요.",
                "개인정보보호": "개인정보보호법의 처리 원칙과 정보주체 권리를 중심으로 분석하세요.",
                "정보보안": "정보보안관리체계(ISMS)의 요구사항과 보안 통제 관점에서 접근하세요.",
                "위험관리": "위험관리 체계의 단계별 절차와 내부통제 관점에서 분석하세요.",
                "정보통신": "정보통신기반 보호법의 보호 요구사항을 중심으로 고려하세요."
            }
            
            if domain in domain_contexts:
                enhanced_context = f"{context_info}\n\n추가 고려사항: {domain_contexts[domain]}"
                return enhanced_context[:800]  # 길이 제한
            
            return context_info[:600]  # 기본 길이 제한
            
        except Exception:
            return context_info if context_info else "관련 법령과 규정을 참고하세요."

    def _is_ratio_question_enhanced(self, question: str, domain: str) -> bool:
        """향상된 비율 질문 확인"""
        question_lower = question.lower()
        
        # 강한 비율 지표
        strong_ratio_indicators = [
            "비율.*얼마", "기준.*비율.*얼마", "정보기술부문.*비율", 
            "예산.*비율", "인력.*비율", ".*%.*이상", "배정.*비율"
        ]
        
        for indicator in strong_ratio_indicators:
            if re.search(indicator, question_lower):
                return True
        
        # 전자금융 도메인의 특별 케이스
        if domain == "전자금융":
            ratio_keywords = ["비율", "기준", "정보기술부문", "예산", "인력", "배정"]
            keyword_count = sum(1 for keyword in ratio_keywords if keyword in question_lower)
            if keyword_count >= 3:
                return True
        
        return False

    def _is_institution_question_enhanced(self, question: str, domain: str) -> bool:
        """향상된 기관 질문 확인"""
        question_lower = question.lower()
        
        # 강한 기관 지표
        institution_indicators = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "분쟁조정.*신청.*기관", "신고.*기관", "상담.*기관", "담당.*기관"
        ]
        
        for indicator in institution_indicators:
            if re.search(indicator, question_lower):
                return True
        
        # 도메인별 기관 관련 키워드
        domain_institution_keywords = {
            "전자금융": ["분쟁조정", "전자금융분쟁조정위원회", "금융감독원"],
            "개인정보보호": ["개인정보보호위원회", "침해신고센터", "신고", "상담"],
            "사이버보안": ["보안관제센터", "사이버보안센터"],
            "정보보안": ["정보보안관리체계", "인증기관"],
            "정보통신": ["과학기술정보통신부"]
        }
        
        if domain in domain_institution_keywords:
            keywords = domain_institution_keywords[domain]
            if any(keyword in question_lower for keyword in keywords) and "기관" in question_lower:
                return True
        
        return False

    def _create_fallback_enhanced_prompt(self, question: str, question_type: str, domain: str, force_diversity: bool) -> str:
        """향상된 폴백 프롬프트 생성"""
        try:
            if question_type == "multiple_choice":
                return f"""다음은 금융보안 관련 객관식 문제입니다. 체계적 분석을 통해 정답을 선택하세요.

문제 분석:
1. 핵심 키워드 파악
2. 선택지별 검토
3. 논리적 추론 적용

문제: {question}

정답 번호: """
            else:
                diversity_note = ""
                if force_diversity:
                    diversity_note = "\n\n중요: 이전과 다른 전문적이고 실무적인 관점에서 답변하세요."
                    
                return f"""다음은 금융보안 관련 주관식 문제입니다. 반드시 한국어로만 전문적이고 정확한 답변을 작성하세요.{diversity_note}

작성 지침:
- 모든 답변은 한국어로만 작성
- 관련 법령과 규정에 근거한 답변
- 구체적이고 실무적인 내용 포함

문제: {question}

한국어 답변: """
                
        except Exception:
            return f"다음 문제에 답변하세요:\n\n{question}\n\n답변: "

    def get_enhanced_context_hints(self, domain: str, intent_type: str) -> str:
        """향상된 도메인별 컨텍스트 힌트"""
        
        enhanced_context_hints = {
            "사이버보안": {
                "특징_묻기": "사이버 위협의 기술적 특성과 동작 방식, 은밀성과 지속성, 공격 벡터를 중심으로 상세히 설명하세요.",
                "지표_묻기": "네트워크 트래픽 이상, 프로세스 활동 패턴, 파일 시스템 변화, 레지스트리 수정 등 구체적인 탐지 지표와 행동 기반 분석 방법을 포함하여 설명하세요.",
                "방안_묻기": "다층 방어체계, 실시간 모니터링, 위협 인텔리전스, 사고 대응 절차, 복구 방안을 포함한 종합적이고 체계적인 대응 방안을 설명하세요."
            },
            "전자금융": {
                "기관_묻기": "전자금융거래법에 근거한 기관의 법적 지위와 구체적 업무 범위, 분쟁조정 절차, 연락처 및 신청 방법을 명확히 설명하세요.",
                "방안_묻기": "접근매체 보안, 거래 기록 보존, 분쟁조정 절차, 이용자 보호 방안을 전자금융거래법 조항과 연계하여 설명하세요.",
                "절차_묻기": "전자금융거래법에 명시된 법적 절차와 당사자별 의무사항, 기한, 방법을 단계별로 상세히 설명하세요.",
                "비율_묻기": "전자금융감독규정 제16조에 명시된 구체적인 수치(5%, 7%)와 법적 근거, 적용 범위, 예외 조건을 포함하여 설명하세요."
            },
            "개인정보보호": {
                "기관_묻기": "개인정보보호법에 따른 기관의 권한과 개인정보 처리 감독 업무, 신고 접수 절차, 분쟁조정 역할을 구체적으로 설명하세요.",
                "방안_묻기": "수집 최소화, 목적 제한, 정보주체 권리 보장 원칙을 중심으로 기술적·관리적·물리적 보호조치를 포함한 종합적 방안을 설명하세요.",
                "절차_묻기": "동의 획득, 처리 현황 공개, 권리 행사 절차를 개인정보보호법 조항에 따라 단계별로 설명하세요."
            },
            "정보보안": {
                "방안_묻기": "정보보안관리체계의 수립, 운영, 점검, 개선 사이클(PDCA)을 중심으로 기술적·관리적·물리적 보안대책을 체계적으로 설명하세요.",
                "요소_묻기": "정보보호의 3대 요소(기밀성, 무결성, 가용성)의 정의와 상호 관계, 각 요소별 보안 통제 방안을 구체적으로 설명하세요.",
                "역할_묻기": "해당 시스템이나 프로토콜의 보안 기능, 인증 메커니즘, 암호화 지원, 위협 대응 능력을 포함한 종합적 역할을 설명하세요."
            },
            "위험관리": {
                "방안_묻기": "위험 식별, 평가, 대응, 모니터링의 4단계 절차와 각 단계별 핵심 활동, 위험 수용 기준, 잔여 위험 관리를 포함하여 설명하세요."
            },
            "정보통신": {
                "방안_묻기": "정보통신기반 보호법에 따른 보고 요구사항과 절차, 응급조치 사항, 관련 기관 협조 체계를 명확히 설명하세요."
            }
        }
        
        try:
            return enhanced_context_hints.get(domain, {}).get(intent_type, 
                "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 체계적으로 설명하세요.")
        except Exception:
            return "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 체계적으로 설명하세요."

    def analyze_enhanced_prompt_effectiveness(self, question: str, answer: str, domain: str) -> Dict:
        """향상된 프롬프트 효과성 분석"""
        try:
            analysis = {
                "answer_length": len(answer) if answer else 0,
                "korean_ratio": self._calculate_korean_ratio_enhanced(answer) if answer else 0,
                "domain_relevance": self._check_domain_relevance_enhanced(answer, domain) if answer else 0,
                "technical_depth": self._assess_technical_depth(answer, domain) if answer else 0,
                "legal_accuracy": self._assess_legal_accuracy(answer, domain) if answer else 0,
                "uniqueness_score": self._calculate_uniqueness_enhanced(answer) if answer else 0,
                "overall_quality": 0
            }
            
            # 종합 품질 점수 계산
            if answer:
                quality_factors = [
                    analysis["korean_ratio"] > 0.8,
                    analysis["answer_length"] > 100,
                    analysis["domain_relevance"] > 0.8,
                    analysis["technical_depth"] > 0.6,
                    analysis["legal_accuracy"] > 0.7,
                    analysis["uniqueness_score"] > 0.6
                ]
                analysis["overall_quality"] = sum(quality_factors) / len(quality_factors)
                
                # 도메인 성능 추적
                if domain not in self.domain_performance:
                    self.domain_performance[domain] = {"count": 0, "avg_quality": 0}
                
                self.domain_performance[domain]["count"] += 1
                prev_avg = self.domain_performance[domain]["avg_quality"]
                count = self.domain_performance[domain]["count"]
                self.domain_performance[domain]["avg_quality"] = (
                    (prev_avg * (count - 1) + analysis["overall_quality"]) / count
                )
            
            return analysis
            
        except Exception as e:
            print(f"향상된 프롬프트 효과성 분석 오류: {e}")
            return {"overall_quality": 0.5}

    def _calculate_korean_ratio_enhanced(self, text: str) -> float:
        """향상된 한국어 비율 계산"""
        if not text:
            return 0.0
        
        try:
            korean_chars = len(re.findall(r"[가-힣]", text))
            meaningful_chars = len(re.sub(r"[^\w가-힣]", "", text))
            return korean_chars / meaningful_chars if meaningful_chars > 0 else 0.0
        except Exception:
            return 0.0

    def _check_domain_relevance_enhanced(self, text: str, domain: str) -> float:
        """향상된 도메인 관련성 확인"""
        if not text or not domain:
            return 0.0
        
        try:
            # 도메인별 핵심 키워드 (가중치 포함)
            domain_keywords = {
                "사이버보안": {
                    "high": ["악성코드", "탐지", "보안위협", "사이버공격", "방어체계"],
                    "medium": ["보안", "위험", "모니터링", "대응", "예방"],
                    "low": ["시스템", "관리", "조치", "방안", "절차"]
                },
                "전자금융": {
                    "high": ["전자금융거래법", "분쟁조정", "전자금융업자", "접근매체"],
                    "medium": ["전자금융", "거래", "보안", "이용자보호"],
                    "low": ["금융", "서비스", "관리", "절차", "방안"]
                },
                "개인정보보호": {
                    "high": ["개인정보보호법", "정보주체", "법정대리인", "개인정보처리"],
                    "medium": ["개인정보", "동의", "처리", "보호", "권리"],
                    "low": ["정보", "관리", "절차", "방안", "조치"]
                }
            }
            
            if domain not in domain_keywords:
                return 0.5
            
            keywords = domain_keywords[domain]
            text_lower = text.lower()
            
            score = 0.0
            total_weight = 0
            
            for level, word_list in keywords.items():
                weight = {"high": 3, "medium": 2, "low": 1}[level]
                matches = sum(1 for word in word_list if word in text_lower)
                score += matches * weight
                total_weight += len(word_list) * weight
            
            return min(score / total_weight, 1.0) if total_weight > 0 else 0.0
            
        except Exception:
            return 0.5

    def _assess_technical_depth(self, text: str, domain: str) -> float:
        """기술적 깊이 평가"""
        if not text:
            return 0.0
        
        try:
            text_lower = text.lower()
            
            # 도메인별 기술 용어
            technical_terms = {
                "사이버보안": ["프로세스", "네트워크", "프로토콜", "암호화", "인증", "로그"],
                "전자금융": ["암호화", "인증", "접근제어", "거래기록", "보안시스템"],
                "개인정보보호": ["암호화", "접근제어", "로그관리", "권한관리"],
                "정보보안": ["관리체계", "통제", "정책", "절차", "감사", "모니터링"]
            }
            
            domain_terms = technical_terms.get(domain, [])
            if not domain_terms:
                return 0.5
            
            matches = sum(1 for term in domain_terms if term in text_lower)
            depth_score = matches / len(domain_terms)
            
            # 설명 깊이 보너스
            if len(text) > 200:
                depth_score += 0.2
            if "따라서" in text or "그러므로" in text or "결과적으로" in text:
                depth_score += 0.1
            
            return min(depth_score, 1.0)
            
        except Exception:
            return 0.5

    def _assess_legal_accuracy(self, text: str, domain: str) -> float:
        """법적 정확성 평가"""
        if not text:
            return 0.0
        
        try:
            text_lower = text.lower()
            
            # 법령 관련 키워드
            legal_keywords = ["법", "조", "항", "규정", "지침", "기준", "요구사항", "의무", "권리"]
            legal_mentions = sum(1 for keyword in legal_keywords if keyword in text_lower)
            
            # 도메인별 주요 법령
            domain_laws = {
                "전자금융": ["전자금융거래법", "전자금융감독규정"],
                "개인정보보호": ["개인정보보호법"],
                "사이버보안": ["정보통신망법", "개인정보보호법"],
                "정보보안": ["정보통신망법", "개인정보보호법"],
                "정보통신": ["정보통신기반보호법"]
            }
            
            domain_law_keywords = domain_laws.get(domain, [])
            law_mentions = sum(1 for law in domain_law_keywords if law in text_lower)
            
            accuracy_score = 0.0
            
            # 기본 법령 언급 점수
            if legal_mentions > 0:
                accuracy_score += 0.4
            
            # 도메인별 법령 언급 점수
            if law_mentions > 0:
                accuracy_score += 0.4
            
            # 구체적 조항 언급 보너스
            if re.search(r'제\d+조', text) or re.search(r'\d+조', text):
                accuracy_score += 0.2
            
            return min(accuracy_score, 1.0)
            
        except Exception:
            return 0.5

    def _calculate_uniqueness_enhanced(self, text: str) -> float:
        """향상된 답변 고유성 계산"""
        if not text:
            return 0.0
        
        try:
            words = text.split()
            if len(words) == 0:
                return 0.0
            
            unique_words = set(words)
            uniqueness_ratio = len(unique_words) / len(words)
            
            # 일반적인 패턴 감지 (확장)
            common_patterns = [
                "에 따라", "해야 합니다", "필요합니다", "관련", "관리", "체계",
                "수립", "시행", "실시", "구축", "운영", "통해", "위해", "대한",
                "이와 같이", "따라서", "그러므로", "또한", "그리고"
            ]
            
            pattern_count = sum(1 for pattern in common_patterns if pattern in text)
            pattern_penalty = min(pattern_count * 0.03, 0.25)  # 페널티 완화
            
            # 반복 구문 감지
            sentences = text.split('.')
            if len(sentences) > 1:
                similar_sentences = 0
                for i in range(len(sentences) - 1):
                    for j in range(i + 1, len(sentences)):
                        if len(sentences[i].strip()) > 10 and len(sentences[j].strip()) > 10:
                            words_i = set(sentences[i].split())
                            words_j = set(sentences[j].split())
                            if words_i and words_j:
                                similarity = len(words_i & words_j) / len(words_i | words_j)
                                if similarity > 0.7:
                                    similar_sentences += 1
                
                repetition_penalty = min(similar_sentences * 0.1, 0.2)
            else:
                repetition_penalty = 0
            
            final_uniqueness = max(uniqueness_ratio - pattern_penalty - repetition_penalty, 0.0)
            return min(final_uniqueness, 1.0)
            
        except Exception:
            return 0.5

    def optimize_prompt_selection(self, domain: str, question_type: str, performance_data: Dict = None) -> Dict:
        """프롬프트 선택 최적화"""
        try:
            optimization_result = {
                "recommended_template": "enhanced",
                "example_count": 2,
                "diversity_level": "medium",
                "focus_areas": [],
                "confidence": 0.8
            }
            
            # 성능 데이터 기반 최적화
            if performance_data:
                domain_perf = performance_data.get(domain, {})
                success_rate = domain_perf.get("success_rate", 0.5)
                avg_quality = domain_perf.get("avg_quality", 0.5)
                
                # 성능이 낮은 도메인은 더 많은 예시와 높은 다양성
                if success_rate < 0.6:
                    optimization_result["example_count"] = 3
                    optimization_result["diversity_level"] = "high"
                    optimization_result["focus_areas"].append("performance_boost")
                
                if avg_quality < 0.7:
                    optimization_result["focus_areas"].append("quality_enhancement")
            
            # 도메인별 최적화
            domain_optimizations = {
                "사이버보안": {
                    "template": "domain_specialized",
                    "example_count": 2,
                    "focus": ["technical_depth", "threat_analysis"]
                },
                "전자금융": {
                    "template": "domain_specialized", 
                    "example_count": 2,
                    "focus": ["legal_accuracy", "procedure_clarity"]
                },
                "개인정보보호": {
                    "template": "domain_specialized",
                    "example_count": 2,
                    "focus": ["legal_compliance", "rights_protection"]
                }
            }
            
            if domain in domain_optimizations:
                domain_opt = domain_optimizations[domain]
                optimization_result.update({
                    "recommended_template": domain_opt["template"],
                    "example_count": domain_opt["example_count"],
                    "focus_areas": domain_opt["focus"]
                })
            
            return optimization_result
            
        except Exception as e:
            print(f"프롬프트 선택 최적화 오류: {e}")
            return {"recommended_template": "enhanced", "example_count": 1, "confidence": 0.5}

    def get_domain_performance_summary(self) -> Dict:
        """도메인별 성능 요약"""
        try:
            summary = {}
            for domain, perf in self.domain_performance.items():
                summary[domain] = {
                    "question_count": perf.get("count", 0),
                    "average_quality": round(perf.get("avg_quality", 0), 3),
                    "performance_trend": "stable"  # 향후 추세 분석 확장 가능
                }
            return summary
        except Exception:
            return {}

    def cleanup(self):
        """리소스 정리"""
        try:
            self.used_examples_cache.clear()
            self.prompt_history.clear()
            self.domain_performance.clear()
        except Exception as e:
            print(f"프롬프트 enhancer 정리 오류: {e}")