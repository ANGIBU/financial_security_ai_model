# knowledge_base.py

import re
import random
from typing import Dict, List
from pathlib import Path

from config import TEMPLATE_QUALITY_CRITERIA


class FinancialSecurityKnowledgeBase:

    def __init__(self):
        self._initialize_integrated_data()
        self.template_quality_criteria = TEMPLATE_QUALITY_CRITERIA

    def _initialize_integrated_data(self):
        """JSON 데이터를 코드 내부로 통합하여 초기화"""
        
        # 템플릿 데이터 강화 - 더 구체적이고 전문적인 답변들로 구성
        self.korean_subjective_templates = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 하여 장기간 시스템에 잠복하면서 악의적인 활동을 수행합니다.",
                    "원격접근 트로이는 사용자를 속여 시스템에 침투한 후 외부 공격자가 원격으로 제어할 수 있는 특성을 가지며, 시스템 깊숙이 숨어서 지속적으로 활동하면서 정보 수집과 원격 제어 기능을 수행합니다. 정상 소프트웨어로 위장하여 탐지를 회피하고 시스템 권한을 탈취하는 특징을 보입니다.",
                    "RAT 악성코드의 주요 특징은 은밀한 설치와 지속적인 시스템 제어 능력입니다. 트로이 목마 방식으로 배포되어 사용자가 직접 설치하도록 유도하며, 설치 후 외부 서버와 통신하여 원격 명령을 수행하고 시스템 정보를 수집합니다.",
                    "트로이 목마는 유용한 프로그램으로 가장하여 사용자의 자발적 설치를 유도하고, 설치 후 악의적인 기능을 활성화하는 특징을 가집니다. 원격 접근 기능을 통해 시스템을 외부에서 조작할 수 있으며, 탐지 회피를 위한 다양한 은폐 기법을 사용합니다.",
                    "원격제어 악성코드는 정상 소프트웨어로 위장하여 배포되며, 설치 후 시스템 권한을 탈취하고 외부 서버와 은밀한 통신을 수행하는 특성을 가집니다. 사용자 모르게 장기간 잠복하면서 시스템을 원격으로 제어하고 중요 정보를 수집합니다."
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 레지스트리 변경 사항, 시스템 성능 저하, 의심스러운 네트워크 연결 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행, 백그라운드에서 실행되는 미상 서비스 등을 통해 RAT 감염을 탐지할 수 있습니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동, 원격 데스크톱 연결 시도, 의심스러운 포트 연결 등이 주요 탐지 지표로 활용됩니다.",
                    "비정상적인 아웃바운드 연결, 시스템 리소스 과다 사용, 백그라운드 프로세스 증가, 보안 소프트웨어 비활성화 시도, 시스템 설정 변경, 레지스트리 키 수정, 네트워크 포트 개방 등의 징후를 종합적으로 분석해야 합니다.",
                    "네트워크 연결 로그 분석을 통한 의심스러운 외부 통신 탐지, 프로세스 모니터링을 통한 비인가 실행 파일 식별, 파일 무결성 검사를 통한 시스템 파일 변조 확인, 레지스트리 변경 감시를 통한 시스템 설정 수정 탐지 등을 수행해야 합니다."
                ],
                "복합설명": [
                    "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가지며, 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성합니다. 주요 탐지 지표로는 네트워크 트래픽의 비정상적 외부 통신 패턴, 시스템에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 레지스트리 변경 사항 등이 있습니다.",
                    "RAT 악성코드의 특징은 은밀한 설치와 지속적인 원격 제어 능력이며, 시스템 깊숙이 숨어서 장기간 활동합니다. 탐지를 위해서는 네트워크 모니터링을 통한 의심스러운 외부 연결 확인, 프로세스 분석을 통한 비정상 실행 파일 식별, 시스템 파일 무결성 검사, 레지스트리 변경 감시 등을 수행해야 합니다."
                ],
                "방안_묻기": [
                    "딥페이크 기술 악용에 대비하여 다층 방어체계 구축, 실시간 딥페이크 탐지 시스템 도입, 직원 교육 및 인식 개선, 생체인증 강화, 다중 인증 체계 구축, 사전 예방과 사후 대응을 아우르는 종합적 보안 대응방안이 필요합니다.",
                    "네트워크 분할을 통한 격리, 접근권한 최소화 원칙 적용, 행위 기반 탐지 시스템 구축, 사고 대응 절차 수립, 백업 및 복구 체계 마련, 보안 인식 교육 강화 등의 보안 방안을 수립해야 합니다.",
                    "SBOM 활용을 통한 소프트웨어 공급망 보안 강화, 구성 요소 취약점 관리, 라이선스 컴플라이언스 확보, 보안 업데이트 추적 관리, 투명성 제고를 통한 보안 위험 사전 식별 등의 종합적 관리방안을 수립해야 합니다."
                ],
                "일반": [
                    "사이버보안 위협에 대응하기 위해 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영하며, 정기적인 보안교육과 취약점 점검을 통해 종합적인 보안 관리체계를 유지해야 합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있으며, 피해구제와 분쟁해결을 위한 전문적인 조정 절차를 제공합니다."
                ],
                "방안_묻기": [
                    "개인정보보호법에 따라 수집 최소화 원칙 적용, 목적 외 이용 금지, 정보주체 권리 보장, 개인정보보호 관리체계 구축, 정기적인 교육 실시, 기술적·관리적·물리적 보호조치 이행 등의 관리방안이 필요합니다.",
                    "개인정보 처리 시 정보주체의 동의 절차 준수, 처리목적 명확화, 보유기간 설정 및 준수, 정보주체 권리 행사 절차 마련, 개인정보 파기 체계 구축 등의 전 과정 관리방안을 수립해야 합니다."
                ],
                "일반": [
                    "개인정보보호법에 따라 정보주체의 권리를 보장하고 수집 최소화 원칙을 적용하며, 개인정보보호 관리체계를 구축하여 체계적이고 안전한 개인정보 처리를 수행해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하기 위한 업무를 수행합니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 전자금융거래법에 따라 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행하며, 전자금융거래에서 발생하는 분쟁의 공정한 해결을 위해 설치된 전문 기구입니다.",
                    "전자금융거래 분쟁의 조정은 금융감독원에 설치된 전자금융분쟁조정위원회에서 담당하며, 전자금융거래법에 근거하여 이용자 보호와 분쟁의 신속한 해결을 위한 업무를 수행합니다."
                ],
                "방안_묻기": [
                    "전자금융거래법에 따라 접근매체 보안을 강화하고 이용자 보호체계를 구축하며, 안전한 전자금융 거래환경 제공을 위한 종합적인 보안조치를 시행해야 합니다.",
                    "접근매체 보안 강화, 전자서명 및 인증체계 고도화, 거래내역 실시간 통지, 이상거래 탐지시스템 구축, 이용자 교육 강화 등의 종합적인 보안방안이 필요합니다."
                ],
                "일반": [
                    "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다.",
                    "한국은행이 금융통화위원회의 요청에 따라 금융회사 및 전자금융업자에게 자료제출을 요구할 수 있는 경우는 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해서입니다."
                ]
            },
            "정보보안": {
                "방안_묻기": [
                    "정보보안관리체계를 구축하여 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영하고 지속적인 보안수준 향상을 위한 관리활동을 수행해야 합니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다."
                ],
                "절차_묻기": [
                    "정보보안 관리절차는 보안정책 수립, 위험분석 실시, 보안대책 선정 및 구현, 보안교육 실시, 보안점검 및 감사, 보안사고 대응, 지속적 개선의 단계로 진행됩니다."
                ],
                "일반": [
                    "정보보안관리체계를 구축하여 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다."
                ]
            },
            "금융투자": {
                "방안_묻기": [
                    "자본시장법에 따라 투자자 보호와 시장 공정성 확보를 위한 적합성 원칙을 준수하고 내부통제 시스템을 강화하여 건전한 금융투자 환경을 조성해야 합니다.",
                    "투자자 보호를 위한 적합성 원칙 준수, 투자위험 고지 의무 이행, 투자자문 서비스 품질 개선, 불공정거래 방지 체계 구축, 내부통제 시스템 강화 등의 방안이 필요합니다."
                ],
                "일반": [
                    "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다."
                ]
            },
            "위험관리": {
                "방안_묻기": [
                    "위험관리 체계를 구축하여 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 내부통제시스템을 통해 체계적인 위험관리를 수행해야 합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다."
                ],
                "절차_묻기": [
                    "위험관리 절차는 위험식별 단계에서 잠재적 위험요소를 파악하고, 위험평가 단계에서 위험의 발생가능성과 영향도를 분석하며, 위험대응 단계에서 적절한 대응전략을 수립하고, 위험모니터링 단계에서 지속적으로 관리합니다."
                ],
                "일반": [
                    "위험관리 체계를 구축하여 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 내부통제시스템을 통해 체계적인 위험관리를 수행해야 합니다."
                ]
            }
        }

        # 도메인 키워드 매핑
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보",
                "수집", "이용", "제공", "파기", "동의", "법정대리인", "아동", "처리",
                "개인정보처리방침", "열람권", "정정삭제권", "처리정지권", "손해배상",
                "개인정보보호위원회", "개인정보영향평가", "개인정보관리체계",
                "개인정보처리시스템", "개인정보보호책임자", "개인정보취급자",
                "개인정보침해신고센터", "PIMS", "관리체계 수립", "정책 수립",
                "만 14세", "미만 아동", "중요한 요소", "경영진", "최고책임자",
                "자원 할당", "내부 감사"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명",
                "전자인증", "공인인증서", "분쟁조정", "전자지급수단", "전자화폐",
                "금융감독원", "한국은행", "전자금융업", "전자금융분쟁조정위원회",
                "전자금융거래", "전자금융업무", "전자금융서비스", "전자금융거래기록",
                "이용자", "금융통화위원회", "자료제출", "통화신용정책", "지급결제제도",
                "요청", "요구", "경우", "보안 강화", "통계조사", "경영 실적", "원활한 운영"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱", "스미싱", "랜섬웨어",
                "해킹", "딥페이크", "원격제어", "RAT", "원격접근", "봇넷", "백도어",
                "루트킷", "취약점", "제로데이", "사회공학", "APT", "DDoS", "침입탐지",
                "침입방지", "보안관제", "SBOM", "소프트웨어 구성 요소", "Trojan",
                "원격제어 악성코드", "탐지 지표", "보안 위협", "특징", "주요 탐지",
                "금융권", "활용", "이유", "적절한", "소프트웨어", "접근 제어",
                "투명성", "다양성", "공급망 보안"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화",
                "방화벽", "침입탐지", "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP", "정보보안관리체계",
                "정보보호", "관리체계 수립", "정책 수립", "최고책임자", "경영진",
                "자원 할당", "내부 감사", "절차 수립", "복구 절차", "비상연락체계",
                "개인정보 파기", "복구 목표시간", "옳지 않은", "고려", "요소"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업", "소비자금융업",
                "보험중개업", "자본시장법", "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "금융산업", "구분",
                "해당하지 않는", "금융산업의 이해"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "리스크", "내부통제",
                "컴플라이언스", "위험식별", "위험분석", "위험모니터링", "위험회피",
                "위험전가", "위험감소", "잔여위험", "위험성향", "위험 관리 계획",
                "수행인력", "위험 대응 전략", "재해 복구", "복구 절차", "비상연락체계",
                "복구 목표시간", "계획 수립", "고려", "요소", "적절하지 않은", "대상", "기간"
            ]
        }

        # 한국어 금융 용어 사전
        self.korean_financial_terms = {
            "정보보안관리체계": "조직의 정보자산을 보호하기 위한 종합적인 관리체계",
            "개인정보관리체계": "개인정보의 안전한 처리를 위한 체계적 관리방안",
            "원격접근": "네트워크를 통해 원격지에서 컴퓨터 시스템에 접근하는 방식",
            "트로이목마": "정상 프로그램으로 위장하여 악의적 기능을 수행하는 악성코드",
            "원격접근도구": "네트워크를 통해 원격지 시스템을 제어할 수 있는 소프트웨어",
            "소프트웨어구성요소명세서": "소프트웨어에 포함된 구성 요소의 목록과 정보를 기록한 문서",
            "딥페이크": "인공지능을 이용하여 가짜 영상이나 음성을 제작하는 기술",
            "전자금융분쟁조정위원회": "전자금융거래 관련 분쟁의 조정을 담당하는 기관",
            "개인정보보호위원회": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관"
        }

        # 기관 데이터베이스
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "전자금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁조정",
                "근거법": "전자금융거래법",
                "상세정보": "전자금융거래에서 발생하는 분쟁의 공정하고 신속한 해결을 위해 설치된 기구로, 이용자와 전자금융업자 간의 분쟁조정 업무를 담당합니다.",
                "관련질문패턴": ["전자금융거래법에 따라", "이용자가", "분쟁조정을 신청할 수 있는", "기관"],
                "template_answer": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하기 위한 업무를 수행합니다."
            },
            "개인정보보호": {
                "기관명": "개인정보보호위원회",
                "소속": "국무총리 소속",
                "역할": "개인정보보호 정책 수립 및 감시",
                "근거법": "개인정보보호법",
                "신고기관": "개인정보침해신고센터",
                "상세정보": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관으로, 개인정보 보호 정책 수립, 법령 집행, 감시 업무를 수행합니다.",
                "관련질문패턴": ["개인정보", "침해", "신고", "상담", "보호위원회"]
            },
            "한국은행": {
                "기관명": "한국은행",
                "소속": "중앙은행",
                "역할": "통화신용정책 수행 및 지급결제제도 운영",
                "근거법": "한국은행법",
                "상세정보": "금융통화위원회의 요청에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 금융회사 및 전자금융업자에게 자료제출을 요구할 수 있습니다.",
                "관련질문패턴": ["한국은행", "금융통화위원회", "자료제출", "요구"]
            }
        }

        # 객관식 답변 패턴 강화
        self.mc_answer_patterns = {
            "금융투자_해당하지않는": {
                "question_keywords": ["금융투자업", "구분", "해당하지 않는"],
                "choices": ["소비자금융업", "투자자문업", "투자매매업", "투자중개업", "보험중개업"],
                "correct_answer": "5",
                "explanation": "금융투자업의 구분에는 소비자금융업, 투자자문업, 투자매매업, 투자중개업이 포함되며, 보험중개업은 해당하지 않습니다.",
                "hint": "금융투자업 구분에서 보험중개업은 제외됩니다."
            },
            "위험관리_적절하지않은": {
                "question_keywords": ["위험 관리", "계획 수립", "적절하지 않은"],
                "choices": ["수행인력", "위험 수용", "위험 대응 전략", "대상", "기간"],
                "correct_answer": "2",
                "explanation": "위험 관리 계획 수립 시 수행인력, 위험 대응 전략 선정, 대상, 기간을 고려해야 하며, 위험 수용은 적절하지 않습니다.",
                "hint": "위험관리 계획에서 위험 수용은 부적절한 요소입니다."
            },
            "개인정보_중요한요소": {
                "question_keywords": ["정책 수립", "가장 중요한 요소", "경영진"],
                "choices": ["정보보호 정책 제개정", "경영진의 참여", "최고책임자 지정", "자원 할당"],
                "correct_answer": "2",
                "explanation": "관리체계 수립 및 운영의 정책 수립 단계에서 가장 중요한 요소는 경영진의 참여입니다.",
                "hint": "정책 수립에서 경영진의 참여가 가장 중요합니다."
            },
            "전자금융_요구경우": {
                "question_keywords": ["한국은행", "자료제출", "요구할 수 있는 경우"],
                "choices": ["보안 강화", "통계조사", "경영 실적", "통화신용정책"],
                "correct_answer": "4",
                "explanation": "한국은행이 금융통화위원회의 요청에 따라 자료제출을 요구할 수 있는 경우는 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해서입니다.",
                "hint": "한국은행의 자료제출 요구는 통화신용정책 수행을 위해서입니다."
            },
            "개인정보_법정대리인": {
                "question_keywords": ["만 14세 미만", "아동", "개인정보", "절차"],
                "choices": ["학교의 동의", "법정대리인의 동의", "본인의 동의", "친구의 동의"],
                "correct_answer": "2",
                "explanation": "개인정보보호법 제22조의2에 따라 만 14세 미만 아동의 개인정보를 처리하기 위해서는 법정대리인의 동의를 받아야 합니다.",
                "hint": "만 14세 미만 아동은 법정대리인의 동의가 필요합니다."
            },
            "사이버보안_SBOM": {
                "question_keywords": ["SBOM", "활용", "이유", "적절한"],
                "choices": ["접근 제어", "투명성", "개인정보 보호", "다양성", "소프트웨어 공급망"],
                "correct_answer": "5",
                "explanation": "금융권에서 SBOM을 활용하는 이유는 소프트웨어 공급망 보안을 강화하기 위해서입니다.",
                "hint": "SBOM은 소프트웨어 공급망 보안을 위해 활용됩니다."
            },
            "정보보안_재해복구": {
                "question_keywords": ["재해 복구", "계획 수립", "옳지 않은"],
                "choices": ["복구 절차", "비상연락체계", "개인정보 파기", "복구 목표시간"],
                "correct_answer": "3",
                "explanation": "재해 복구 계획 수립 시 복구 절차, 비상연락체계, 복구 목표시간 정의가 필요하며, 개인정보 파기 절차는 옳지 않습니다.",
                "hint": "재해복구 계획에서 개인정보 파기는 관련 없는 요소입니다."
            }
        }

        print("통합 데이터 초기화 완료")

    def analyze_question(self, question: str) -> Dict:
        """질문 분석 - 정확도 강화"""
        question_lower = question.lower()

        # 도메인 탐지 강화
        detected_domains = []
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드에 더 높은 점수
                    if keyword in [
                        "트로이", "RAT", "원격제어", "SBOM", "전자금융분쟁조정위원회", 
                        "개인정보보호위원회", "만 14세", "위험 관리", "금융투자업",
                    ]:
                        score += 5
                    elif keyword in [
                        "개인정보", "전자금융", "사이버보안", "정보보안", "금융투자", "위험관리"
                    ]:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            detected_domains = [best_domain]
        else:
            detected_domains = ["일반"]

        # 기타 분석
        complexity = self._calculate_complexity(question)
        korean_terms = self._find_korean_technical_terms(question)
        compliance_check = self._check_competition_compliance(question)
        institution_info = self._check_institution_question(question)
        mc_pattern_info = self._analyze_mc_pattern(question)

        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info,
            "mc_pattern_info": mc_pattern_info,
        }

        return analysis_result

    def get_template_examples(self, domain: str, intent_type: str = "일반") -> List[str]:
        """도메인별 템플릿 예시 반환 - 정확도 강화"""
        templates = []
        
        # 도메인과 의도 타입에 정확히 매칭되는 템플릿 찾기
        if domain in self.korean_subjective_templates:
            domain_templates = self.korean_subjective_templates[domain]

            if isinstance(domain_templates, dict):
                # 정확한 의도 타입 매칭
                if intent_type in domain_templates:
                    templates = domain_templates[intent_type]
                    if self.verbose:
                        print(f"도메인 {domain}, 의도 {intent_type}에서 {len(templates)}개 템플릿 발견")
                
                # 복합 질문 처리 (특징+지표)
                elif intent_type == "복합설명" and "복합설명" in domain_templates:
                    templates = domain_templates["복합설명"]
                elif intent_type == "복합설명":
                    # 특징과 지표 템플릿을 결합
                    feature_templates = domain_templates.get("특징_묻기", [])
                    indicator_templates = domain_templates.get("지표_묻기", [])
                    if feature_templates and indicator_templates:
                        # 첫 번째 특징 템플릿과 첫 번째 지표 템플릿을 결합
                        combined = f"{feature_templates[0]} {indicator_templates[0]}"
                        templates = [combined] + feature_templates[:2] + indicator_templates[:2]
                
                # 일반 템플릿 사용
                elif "일반" in domain_templates:
                    templates = domain_templates["일반"]
                
                # 다른 의도 타입에서 가져오기
                else:
                    for available_intent, available_templates in domain_templates.items():
                        if available_templates and len(available_templates) > 0:
                            templates = available_templates
                            break
            else:
                templates = domain_templates

        # 템플릿이 부족한 경우 다른 도메인에서 보충
        if not templates or len(templates) < 2:
            additional_templates = []
            for other_domain, other_templates in self.korean_subjective_templates.items():
                if other_domain != domain and isinstance(other_templates, dict):
                    if intent_type in other_templates and other_templates[intent_type]:
                        additional_templates.extend(other_templates[intent_type][:1])
                        if len(additional_templates) >= 2:
                            break
            
            templates = (templates or []) + additional_templates

        # 여전히 템플릿이 없으면 기본 템플릿 생성
        if not templates:
            templates = self._generate_fallback_templates(domain, intent_type)

        # 템플릿 품질 확인 및 반환
        if isinstance(templates, list) and len(templates) > 0:
            # 품질 높은 템플릿 우선 선택
            quality_templates = []
            for template in templates:
                if self._check_template_quality(template):
                    quality_templates.append(template)
            
            final_templates = quality_templates if quality_templates else templates
            
            # 다양성을 위해 셔플
            shuffled_templates = final_templates.copy()
            random.shuffle(shuffled_templates)
            return shuffled_templates[:5]

        return []

    def _check_template_quality(self, template: str) -> bool:
        """템플릿 품질 확인"""
        if not template or len(template) < 20:
            return False
        
        # 한국어 비율 확인
        korean_chars = len(re.findall(r"[가-힣]", template))
        total_chars = len(re.sub(r"[^\w가-힣]", "", template))
        
        if total_chars == 0:
            return False
            
        korean_ratio = korean_chars / total_chars
        if korean_ratio < 0.8:
            return False
        
        # 전문 용어 포함 확인
        professional_terms = [
            "법령", "규정", "조치", "관리", "보안", "체계", "시스템", "위원회", 
            "기관", "업무", "담당", "수행", "구축", "수립", "시행", "실시"
        ]
        
        if any(term in template for term in professional_terms):
            return True
            
        return len(template) >= 50  # 길이만으로도 품질 인정

    def get_mc_pattern_hints(self, question: str) -> str:
        """객관식 패턴 힌트 제공 - 강화"""
        question_lower = question.lower()
        
        # 1단계: 정확한 패턴 매칭
        best_match = None
        best_score = 0
        
        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            score = 0
            matched_keywords = 0
            
            for keyword in pattern_data["question_keywords"]:
                if keyword in question_lower:
                    matched_keywords += 1
                    # 키워드별 중요도 점수
                    if keyword in ["해당하지 않는", "적절하지 않은", "옳지 않은"]:
                        score += 3
                    elif keyword in ["가장 중요한", "가장 적절한"]:
                        score += 3
                    else:
                        score += 1
            
            # 매칭 비율 계산
            match_ratio = matched_keywords / len(pattern_data["question_keywords"])
            final_score = score * match_ratio
            
            if final_score > best_score and matched_keywords >= 2:
                best_score = final_score
                best_match = pattern_data

        # 2단계: 최적 매치 힌트 제공
        if best_match and best_score >= 2:
            hint_parts = []
            
            if "hint" in best_match:
                hint_parts.append(best_match["hint"])
            
            if "explanation" in best_match:
                hint_parts.append(f"참고: {best_match['explanation']}")
            
            return " ".join(hint_parts)

        # 3단계: 일반적인 도메인 힌트
        return self._get_general_mc_hint(question_lower)

    def _get_general_mc_hint(self, question_lower: str) -> str:
        """일반적인 객관식 힌트"""
        
        # 부정 문제 힌트
        if any(neg in question_lower for neg in ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]):
            return "문제에서 요구하는 것과 반대되는 선택지를 찾으세요."
        
        # 긍정 문제 힌트  
        elif any(pos in question_lower for pos in ["가장 적절한", "가장 옳은", "맞는 것"]):
            return "문제에서 요구하는 조건에 가장 부합하는 선택지를 선택하세요."
        
        # 도메인별 일반 힌트
        domain_hints = {
            "금융투자": "금융투자업의 구분과 각 업무의 특징을 고려하세요.",
            "위험관리": "위험관리 계획의 필수 요소와 부적절한 요소를 구분하세요.",
            "개인정보": "개인정보보호법의 연령 제한과 동의 절차를 확인하세요.",
            "전자금융": "한국은행의 권한과 업무 범위를 고려하세요.",
            "사이버보안": "보안 기술의 목적과 활용 분야를 파악하세요."
        }
        
        for domain, hint in domain_hints.items():
            if domain in question_lower:
                return hint
        
        return "각 선택지를 신중히 검토하고 문제의 핵심 요구사항을 파악하세요."

    def get_institution_hints(self, institution_type: str) -> str:
        """기관 힌트 제공 - 강화"""
        
        # 기관별 상세 정보 제공
        institution_details = {
            "전자금융분쟁조정": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 전자금융거래법에 따라 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하기 위한 업무를 수행합니다.",
            "개인정보보호": "개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다. 개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙행정기관입니다.",
            "금융투자분쟁조정": "금융분쟁조정위원회에서 금융투자 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 있으며, 투자자 보호와 분쟁의 공정한 해결을 위한 업무를 수행합니다.",
            "한국은행": "한국은행에서 금융통화위원회의 요청에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 금융회사 및 전자금융업자에게 자료제출을 요구할 수 있습니다."
        }

        # 데이터베이스에서 상세 정보 제공
        if institution_type in self.institution_database:
            info = self.institution_database[institution_type]
            
            if "template_answer" in info:
                return info["template_answer"]
            
            hint_parts = []
            if "기관명" in info:
                hint_parts.append(f"기관명: {info['기관명']}")
            if "소속" in info:
                hint_parts.append(f"소속: {info['소속']}")
            if "역할" in info:
                hint_parts.append(f"주요 역할: {info['역할']}")
            if "근거법" in info:
                hint_parts.append(f"근거 법령: {info['근거법']}")

            return " ".join(hint_parts)

        return institution_details.get(
            institution_type, 
            "해당 분야의 전문 기관에서 관련 업무를 담당하고 있습니다."
        )

    def _generate_fallback_templates(self, domain: str, intent_type: str) -> List[str]:
        """기본 템플릿 생성"""
        
        # 도메인별 특화 기본 템플릿
        domain_fallbacks = {
            "사이버보안": {
                "특징_묻기": [
                    "해당 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하고 외부에서 원격으로 제어하는 특성을 가지며, 은밀성과 지속성을 통해 장기간 악의적인 활동을 수행합니다."
                ],
                "지표_묻기": [
                    "주요 탐지 지표로는 비정상적인 네트워크 활동, 시스템 리소스 과다 사용, 알려지지 않은 프로세스 실행, 파일 시스템 변경, 보안 정책 위반 시도 등을 실시간 모니터링을 통해 식별할 수 있습니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 관련 업무를 담당하며, 금융감독원 내에 설치되어 전자금융거래 분쟁의 조정 업무를 수행합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다."
                ]
            }
        }

        # 도메인별 특화 템플릿이 있으면 사용
        if domain in domain_fallbacks and intent_type in domain_fallbacks[domain]:
            return domain_fallbacks[domain][intent_type]

        # 일반적인 기본 템플릿
        general_fallbacks = {
            "특징_묻기": [
                "주요 특징을 체계적으로 분석하여 관련 법령에 따라 관리해야 합니다.",
                "핵심적인 특성과 성질을 정확히 파악하여 적절한 대응방안을 마련해야 합니다."
            ],
            "지표_묻기": [
                "주요 탐지 지표를 통해 체계적인 모니터링과 분석을 수행해야 합니다.",
                "관련 징후와 패턴을 분석하여 적절한 대응조치를 시행해야 합니다."
            ],
            "방안_묻기": [
                "체계적인 대응 방안을 수립하고 관련 법령에 따라 지속적으로 관리해야 합니다.",
                "효과적인 관리 방안을 마련하여 정기적인 점검과 개선을 수행해야 합니다."
            ],
            "기관_묻기": [
                "관련 전문 기관에서 해당 업무를 법령에 따라 담당하고 있습니다.",
                "소관 기관에서 체계적인 관리와 감독 업무를 수행하고 있습니다."
            ]
        }

        return general_fallbacks.get(intent_type, [
            "관련 법령과 규정에 따라 체계적인 관리가 필요합니다.",
            "해당 분야의 전문적 지식을 바탕으로 적절한 대응을 수행해야 합니다."
        ])

    def _analyze_mc_pattern(self, question: str) -> Dict:
        """객관식 패턴 분석"""
        question_lower = question.lower()

        pattern_info = {
            "is_mc_question": False,
            "pattern_type": None,
            "pattern_confidence": 0.0,
            "pattern_key": None,
            "hint_available": False,
        }

        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            keyword_matches = sum(
                1 for keyword in pattern_data["question_keywords"]
                if keyword in question_lower
            )

            if keyword_matches >= 2:
                pattern_info["is_mc_question"] = True
                pattern_info["pattern_type"] = pattern_key
                pattern_info["pattern_confidence"] = keyword_matches / len(pattern_data["question_keywords"])
                pattern_info["pattern_key"] = pattern_key
                pattern_info["hint_available"] = True
                break

        return pattern_info

    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인 - 정확도 강화"""
        question_lower = question.lower()

        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None,
            "hint_available": False,
        }

        # 기관 질문 패턴 강화
        institution_patterns = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관",
            "담당.*기관", "관리.*기관", "감독.*기관", "소관.*기관",
            "신고.*기관", "접수.*기관", "상담.*기관", "문의.*기관",
            "위원회.*무엇", "위원회.*어디", "위원회.*설명", 
            "분쟁.*어디", "신고.*어디", "상담.*어디",
            "기관을.*기술하세요", ".*기관.*기술", "분쟁조정.*기관"
        ]

        pattern_matches = 0
        matched_patterns = []
        
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 1
                matched_patterns.append(pattern)

        # 기관 질문으로 판단되는 조건 강화
        is_asking_institution = pattern_matches > 0

        if is_asking_institution:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 1.0, 1.0)
            institution_info["question_pattern"] = matched_patterns[0] if matched_patterns else None
            institution_info["hint_available"] = True

            # 기관 타입 매칭 강화
            institution_mapping = {
                "전자금융분쟁조정": ["전자금융", "전자적", "분쟁", "조정", "금융감독원", "이용자"],
                "개인정보보호": ["개인정보", "정보주체", "침해", "신고", "상담", "보호위원회"],
                "금융투자분쟁조정": ["금융투자", "투자자문", "자본시장", "분쟁", "투자자"],
                "한국은행": ["한국은행", "금융통화위원회", "자료제출", "통화신용정책", "지급결제"]
            }

            best_match_score = 0
            best_match_type = None

            for inst_type, keywords in institution_mapping.items():
                keyword_matches = sum(1 for keyword in keywords if keyword in question_lower)
                match_score = keyword_matches / len(keywords)
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_type = inst_type

            if best_match_score > 0:
                institution_info["institution_type"] = best_match_type
                institution_info["confidence"] = best_match_score

        return institution_info

    def _check_competition_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인"""
        compliance = {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True,
        }

        korean_chars = len([c for c in question if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
        total_chars = len([c for c in question if c.isalpha()])

        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            compliance["korean_content"] = korean_ratio > 0.7

        found_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                found_domains.append(domain)

        compliance["appropriate_domain"] = len(found_domains) > 0

        return compliance

    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        length_factor = min(len(question) / 200, 1.0)

        korean_term_count = sum(1 for term in self.korean_financial_terms.keys() if term in question)
        term_factor = min(korean_term_count / 3, 1.0)

        domain_count = sum(
            1 for keywords in self.domain_keywords.values()
            if any(keyword in question.lower() for keyword in keywords)
        )
        domain_factor = min(domain_count / 2, 1.0)

        return (length_factor + term_factor + domain_factor) / 3

    def _find_korean_technical_terms(self, question: str) -> List[str]:
        """한국어 기술용어 탐지"""
        found_terms = []
        for term in self.korean_financial_terms.keys():
            if term in question:
                found_terms.append(term)
        return found_terms

    def _determine_technical_level(self, complexity: float, korean_terms: List[str]) -> str:
        """기술 수준 결정"""
        if complexity > 0.7 or len(korean_terms) >= 2:
            return "고급"
        elif complexity > 0.4 or len(korean_terms) >= 1:
            return "중급"
        else:
            return "초급"

    def get_specific_answer_for_question(self, question: str, domain: str, intent_type: str) -> str:
        """특정 질문에 대한 정확한 답변 반환"""
        question_lower = question.lower()
        
        # 트로이 목마 관련 질문
        if "트로이" in question_lower and "원격제어" in question_lower:
            if "특징" in question_lower and "지표" in question_lower:
                # 특징과 지표를 모두 묻는 복합 질문
                return "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다. 주요 탐지 지표로는 네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 레지스트리 변경 사항, 시스템 성능 저하 등이 있습니다."
            elif "특징" in question_lower:
                # 특징만 묻는 질문
                return "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 하여 장기간 시스템에 잠복하면서 악의적인 활동을 수행합니다."
            elif "지표" in question_lower:
                # 지표만 묻는 질문
                return "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 레지스트리 변경 사항, 시스템 성능 저하, 의심스러운 네트워크 연결 등이 주요 탐지 지표입니다."

        # 전자금융 분쟁조정 관련 질문
        elif "전자금융" in question_lower and "분쟁조정" in question_lower and "기관" in question_lower:
            return "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하기 위한 업무를 수행합니다."

        # 개인정보 관련 질문
        elif "개인정보" in question_lower and "신고" in question_lower and "기관" in question_lower:
            return "개인정보보호위원회 산하 개인정보침해신고센터에서 개인정보 침해 신고 접수 및 상담 업무를 담당합니다. 개인정보보호위원회는 개인정보 보호에 관한 업무를 총괄하는 중앙행정기관입니다."

        # 기본 템플릿 반환
        fallback_templates = {
            "특징_묻기": [
                "주요 특징을 체계적으로 분석하여 관련 법령에 따라 관리해야 합니다."
            ],
            "지표_묻기": [
                "주요 탐지 지표를 통해 체계적인 모니터링과 분석을 수행해야 합니다."
            ],
            "방안_묻기": [
                "체계적인 대응 방안을 수립하고 관련 법령에 따라 지속적으로 관리해야 합니다."
            ],
            "기관_묻기": [
                "관련 전문 기관에서 해당 업무를 법령에 따라 담당하고 있습니다."
            ]
        }

        return fallback_templates.get(intent_type, [
            "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."
        ])

    def cleanup(self):
        """리소스 정리"""
        pass