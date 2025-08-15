# knowledge_base.py

"""
금융보안 지식베이스 (성능 최적화)
- 도메인별 키워드 분류 최적화
- 전문 용어 처리 고도화
- 한국어 전용 답변 템플릿 제공 (품질 개선)
- 대회 규칙 준수 검증 강화
- 질문 의도별 지식 제공 고도화
- 고품질 템플릿 관리 시스템 최적화
- 캐싱 및 성능 최적화
"""

import pickle
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Set
from pathlib import Path
import random
import hashlib
from collections import defaultdict, LRU

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스 (성능 최적화)"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 성능 최적화를 위한 캐시 시스템
        self.domain_cache = {}
        self.pattern_cache = {}
        self.template_cache = {}
        self.keyword_trie = {}
        
        # 도메인별 키워드 (CSV 분석 기반 최적화)
        self.domain_keywords = {
            "전자금융": [
                "전자금융", "전자서명", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서", "전자금융업",
                "전자지급수단", "전자화폐", "전자금융거래", "인증",
                "전자금융분쟁조정위원회", "금융감독원", "한국은행",
                "전자금융거래기록", "전자금융업무", "전자적장치",
                "전자금융거래약관", "전자금융서비스", "전자금융업무위탁",
                "접근매체위조", "접근매체변조", "접근매체도용",
                "전자금융거래오류", "전자금융거래분쟁", "손해배상",
                "이용자보호", "분쟁조정", "피해구제", "보안조치",
                "전자금융업신고", "전자금융업등록", "전자금융업인가",
                "비대면거래", "본인확인", "거래한도", "이상거래탐지"
            ],
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "수집", "이용", "제공", "파기", "동의",
                "법정대리인", "아동", "처리", "개인정보처리방침", "열람권",
                "정정삭제권", "처리정지권", "손해배상", "개인정보보호위원회",
                "개인정보영향평가", "개인정보관리체계", "개인정보처리시스템",
                "개인정보보호책임자", "개인정보취급자", "개인정보침해신고센터",
                "가명정보", "익명정보", "결합", "비식별조치", "재식별",
                "정보주체권리", "개인정보이용내역", "개인정보수집현황",
                "개인정보처리현황", "개인정보보호수준", "개인정보침해",
                "개인정보유출", "개인정보오남용", "개인정보도용",
                "처리목적", "보유기간", "제3자제공", "위탁처리"
            ],
            "사이버보안": [
                "트로이", "악성코드", "해킹", "멀웨어", "피싱", 
                "스미싱", "랜섬웨어", "바이러스", "웜", "스파이웨어",
                "원격제어", "원격접근", "RAT", "봇넷", "분산서비스거부공격", 
                "지능형지속위협", "제로데이", "딥페이크", "사회공학", 
                "취약점", "패치", "침입탐지", "침입방지", "보안관제",
                "백도어", "루트킷", "키로거", "트로이목마", "원격접근도구",
                "APT", "DDoS", "SQL인젝션", "XSS", "CSRF",
                "버퍼오버플로우", "패스워드크래킹", "사전공격", "무차별공격",
                "중간자공격", "DNS스푸핑", "ARP스푸핑", "세션하이재킹",
                "크리덴셜스터핑", "패스워드스프레이", "브루트포스"
            ],
            "정보보안": [
                "정보보안", "보안관리", "정보보안관리체계", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지시스템",
                "침입방지시스템", "보안정보이벤트관리", "보안관제", "인증",
                "권한관리", "로그관리", "백업", "복구", "재해복구",
                "비즈니스연속성계획", "보안감사", "보안교육", "ISMS",
                "ISMS-P", "ISO27001", "CC", "보안통제", "위험관리",
                "보안사고", "사고대응", "CERT", "CSIRT", "SOC",
                "SIEM", "DLP", "NAC", "VPN", "PKI",
                "디지털포렌식", "보안컨설팅", "보안진단", "모의해킹",
                "취약점진단", "보안성검토", "보안인증", "보안제품"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "집합투자업", "신탁업", "소비자금융업", "보험중개업",
                "금융투자회사", "자본시장법", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "투자권유",
                "금융투자상품", "투자위험", "투자성과", "수익률",
                "투자손실", "투자설명서", "투자위험고지서", "투자계약서",
                "투자자문계약", "투자일임계약", "집합투자계약", "신탁계약",
                "펀드운용", "자산운용", "포트폴리오", "리스크관리",
                "파생상품거래", "선물거래", "옵션거래", "스왑거래"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "위험회피",
                "위험전가", "위험감소", "위험분석", "위험식별", "위험모니터링",
                "리스크", "내부통제", "컴플라이언스", "감사", "위험통제",
                "위험보고", "위험문화", "위험거버넌스", "위험한도",
                "신용위험", "시장위험", "운영위험", "유동성위험", "금리위험",
                "환율위험", "집중위험", "명성위험", "전략위험", "규제위험",
                "기술위험", "사이버위험", "모델위험", "컨덕트위험",
                "ESG위험", "기후위험", "지정학적위험", "팬데믹위험",
                "위험측정", "위험계량", "스트레스테스트", "시나리오분석"
            ]
        }
        
        # 키워드 Trie 구조 생성 (검색 성능 최적화)
        self._build_keyword_trie()
        
        # 객관식 질문 패턴 (강화)
        self.mc_patterns = [
            "해당하지.*않는.*것",
            "적절하지.*않는.*것", 
            "옳지.*않는.*것",
            "틀린.*것",
            "맞는.*것",
            "옳은.*것",
            "적절한.*것",
            "올바른.*것",
            "가장.*적절한.*것",
            "가장.*옳은.*것"
        ]
        
        # 한국어 전용 주관식 답변 템플릿 (CSV 기반 최적화)
        self.korean_subjective_templates = {
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 전자금융거래법에 따라 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다. 전자금융거래에서 발생하는 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결합니다.",
                    "전자금융거래법에 따라 금융감독원의 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁의 조정 업무를 담당하고 있습니다. 이용자 보호와 금융시장 안정을 위한 분쟁해결 기구로 기능합니다.",
                    "전자금융 분쟁조정은 금융감독원에 설치된 전자금융분쟁조정위원회에서 신청할 수 있으며, 이용자 보호를 위한 분쟁해결 업무를 수행합니다. 무료로 분쟁조정 서비스를 제공하여 이용자의 권익을 보호합니다."
                ],
                "일반": [
                    "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다. 이용자 보호와 건전한 전자금융시장 발전을 위한 제도적 기반을 제공합니다.",
                    "전자금융업자는 접근매체의 위조나 변조를 방지하기 위한 대책을 강구하고 이용자에게 안전한 거래환경을 제공해야 합니다. 전자금융거래 시 보안프로토콜 준수와 이용자 인증이 필수적입니다.",
                    "전자금융거래의 무결성과 기밀성 보장을 위해 강력한 암호화 기술 적용, 접근통제 시스템 구축, 거래로그 관리, 보안감사 실시 등의 종합적인 보안관리가 필요합니다. 디지털 금융 혁신과 보안의 균형이 중요합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보 침해신고센터에서 신고 접수 및 상담 업무를 담당합니다. 개인정보 관련 분쟁의 조정은 개인정보보호위원회 내 개인정보 분쟁조정위원회에서 담당하며, 피해구제와 분쟁해결 업무를 수행합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다. 개인정보 처리자에 대한 조사와 제재 권한도 가지고 있습니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있습니다. 온라인과 오프라인을 통한 다양한 신고 접수 채널을 운영하며 24시간 신고 접수가 가능합니다."
                ],
                "일반": [
                    "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다. 개인정보 처리의 적법성과 정당성을 확보하고 정보주체의 동의를 받아야 합니다.",
                    "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다. 개인정보 침해 시 즉시 신고하고 피해 최소화 조치를 취해야 합니다.",
                    "개인정보 수집 시 수집목적과 이용범위를 명확히 고지하고 정보주체의 명시적 동의를 받아야 하며, 수집된 개인정보는 목적 달성 후 지체없이 파기해야 합니다. 개인정보 처리 방침을 수립하고 공개해야 합니다."
                ]
            },
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격접근도구는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다. 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다. 탐지를 회피하기 위한 다양한 기법을 사용하며 지속적인 위협을 가합니다.",
                    "원격접근 도구의 주요 특징은 은밀한 설치, 지속적인 연결 유지, 시스템 전반에 대한 제어권 획득, 사용자 모르게 정보 수집 등이며, 탐지를 회피하기 위한 다양한 기법을 사용합니다. 자동 실행과 은닉 기능을 통해 장기간 시스템에 잠복합니다."
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다. 시스템 성능 저하와 예상치 못한 네트워크 활동도 중요한 지표가 됩니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다. 로그 분석과 행위 기반 탐지를 통한 이상 징후 식별이 중요합니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다. DNS 쿼리 패턴과 통신 프로토콜 분석도 중요한 지표입니다."
                ],
                "일반": [
                    "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다. 보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다.",
                    "악성코드 탐지를 위한 행위 기반 분석과 시그니처 기반 탐지를 병행하고, 네트워크 트래픽 모니터링을 통해 이상 징후를 조기에 발견해야 합니다. 제로데이 공격 대응과 APT 공격 탐지 역량 강화가 중요합니다.",
                    "사이버 보안 체계는 예방, 탐지, 대응, 복구의 단계별 접근을 통해 구축되어야 하며, 각 단계별로 적절한 보안 통제와 기술적 대응 방안을 마련해야 합니다. 보안 거버넌스와 컴플라이언스 준수가 필수적입니다."
                ]
            },
            "정보보안": {
                "방안_묻기": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다. 보안 거버넌스 체계 구축과 최고경영진의 보안 의지가 중요합니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다. 제로트러스트 보안 모델 적용과 지속적인 보안성 검증이 필요합니다.",
                    "정보자산 분류체계를 구축하고 중요도에 따른 차등 보안조치를 적용하며, 정기적인 보안교육과 인식제고 프로그램을 운영해야 합니다. 클라우드 보안과 모바일 보안 강화도 필수적입니다."
                ],
                "일반": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다. 보안 거버넌스와 최고경영진의 의지가 성공의 핵심요소입니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다. 보안 위협의 진화에 대응한 동적 보안 체계 구축이 필요합니다."
                ]
            },
            "금융투자": {
                "일반": [
                    "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다. 투자자의 이익을 최우선으로 하는 수탁자 책임을 이행해야 합니다.",
                    "금융투자업 영위 시 투자자의 투자성향과 위험도를 평가하고 적합한 상품을 권유하는 적합성 원칙을 준수해야 합니다. 투자 설명서 제공과 투자위험 고지가 의무사항입니다.",
                    "투자자문업자는 고객의 투자목적과 재정상황을 종합적으로 고려하여 적절한 투자자문을 제공하고 이해상충을 방지해야 합니다. 투자자의 최선의 이익을 위한 선관주의 의무를 부담합니다."
                ]
            },
            "위험관리": {
                "방안_묻기": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다. 통합위험관리 체계와 스트레스 테스트 정기 실시가 중요합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다. 신종 위험과 복합 위험에 대한 대응 체계 구축이 필요합니다.",
                    "위험관리 정책과 절차를 수립하고 위험한도를 설정하여 관리하며, 위험관리 조직과 책임체계를 명확히 정의해야 합니다. 위험 문화 조성과 위험 커뮤니케이션 강화가 중요합니다."
                ],
                "일반": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다. 전사적 위험관리 관점에서 통합적 접근이 필요합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다. 위험 기반 내부통제와 3선 방어 체계 구축이 효과적입니다."
                ]
            },
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다. 전문적인 접근과 종합적인 관리체계 구축이 필요합니다.",
                    "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다. 위험 기반 접근법과 예방 중심의 관리가 효과적입니다.",
                    "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다. 이해관계자 참여와 지속적인 개선 활동이 중요합니다.",
                    "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다. 예방과 대응의 균형잡힌 접근이 필요합니다.",
                    "조직의 정책과 절차에 따라 업무를 수행하고 지속적인 개선활동을 실시해야 합니다. 성과 측정과 환류 체계 구축이 효과적입니다.",
                    "해당 분야의 전문기관과 협력하여 체계적인 관리체계를 구축하고 운영해야 합니다. 모범사례 벤치마킹과 전문성 강화가 중요합니다.",
                    "국제 표준과 모범사례를 참조하여 관리체계를 구축하고, 정기적인 성과 평가와 지속적인 개선을 통해 관리 수준을 향상시켜야 합니다."
                ]
            }
        }
        
        # 템플릿 품질 평가 기준 (강화)
        self.template_quality_criteria = {
            "length_range": (80, 450),
            "korean_ratio_min": 0.95,
            "structure_keywords": ["법", "규정", "조치", "관리", "절차", "기준", "체계", "정책"],
            "intent_keywords": {
                "기관_묻기": ["위원회", "기관", "담당", "업무", "조정", "분쟁", "신고", "접수"],
                "특징_묻기": ["특징", "특성", "성질", "기능", "속성", "성격", "원리", "메커니즘"],
                "지표_묻기": ["지표", "징후", "패턴", "탐지", "신호", "증상", "단서", "흔적"],
                "방안_묻기": ["방안", "대책", "조치", "관리", "대응", "해결", "개선", "강화"],
                "절차_묻기": ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행", "처리"],
                "조치_묻기": ["조치", "대응", "대책", "방안", "보안", "예방", "개선", "보완"],
                "법령_묻기": ["법", "법령", "법률", "규정", "조항", "규칙", "기준", "근거"],
                "정의_묻기": ["정의", "개념", "의미", "뜻", "용어", "설명", "해석", "이해"]
            },
            "professional_terms": [
                "금융보안", "정보보안", "개인정보보호", "전자금융", "사이버보안",
                "위험관리", "내부통제", "컴플라이언스", "보안관제", "침입탐지"
            ],
            "quality_indicators": [
                "체계적", "종합적", "구체적", "전문적", "효과적", "지속적", "선제적", "통합적"
            ]
        }
        
        # 질문 분석 이력
        self.analysis_history = {
            "domain_frequency": defaultdict(int),
            "complexity_distribution": {},
            "question_patterns": [],
            "compliance_check": {
                "korean_only": 0,
                "law_references": 0,
                "technical_terms": 0
            },
            "intent_analysis_history": defaultdict(list),
            "template_usage_stats": defaultdict(int),
            "template_effectiveness": defaultdict(list),
            "high_quality_template_bank": defaultdict(list),
            "domain_expertise_score": defaultdict(float),
            "answer_quality_trends": [],
            "best_practice_patterns": defaultdict(list),
            "pattern_cache_stats": {"hits": 0, "misses": 0}
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            "cache_hit_rate": 0.0,
            "avg_response_time": 0.0,
            "template_selection_time": 0.0,
            "domain_detection_time": 0.0
        }
        
        # 이전 분석 이력 로드
        self._load_analysis_history()
    
    def _build_keyword_trie(self):
        """키워드 Trie 구조 생성 (검색 성능 최적화)"""
        for domain, keywords in self.domain_keywords.items():
            self.keyword_trie[domain] = {}
            for keyword in keywords:
                current = self.keyword_trie[domain]
                for char in keyword:
                    if char not in current:
                        current[char] = {}
                    current = current[char]
                current['$'] = True  # 단어 종료 마커
    
    def _search_keywords_fast(self, text: str, domain: str) -> List[str]:
        """Trie를 사용한 빠른 키워드 검색"""
        found_keywords = []
        trie = self.keyword_trie.get(domain, {})
        
        for i in range(len(text)):
            current = trie
            j = i
            word = ""
            
            while j < len(text) and text[j] in current:
                word += text[j]
                current = current[text[j]]
                
                if '$' in current:  # 완전한 단어 발견
                    found_keywords.append(word)
                
                j += 1
        
        return found_keywords
    
    def _load_analysis_history(self):
        """이전 분석 이력 로드"""
        history_file = self.pkl_dir / "analysis_history.pkl"
        
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    saved_history = pickle.load(f)
                    for key, value in saved_history.items():
                        if key in self.analysis_history:
                            if isinstance(self.analysis_history[key], defaultdict):
                                self.analysis_history[key].update(value)
                            else:
                                self.analysis_history[key] = value
            except Exception:
                pass
    
    def _save_analysis_history(self):
        """분석 이력 저장"""
        history_file = self.pkl_dir / "analysis_history.pkl"
        
        try:
            save_data = {}
            for key, value in self.analysis_history.items():
                if isinstance(value, defaultdict):
                    save_data[key] = dict(value)
                else:
                    save_data[key] = value
            
            save_data["last_updated"] = datetime.now().isoformat()
            
            # 최근 데이터만 저장 (메모리 최적화)
            if "question_patterns" in save_data:
                save_data["question_patterns"] = save_data["question_patterns"][-2000:]
            if "answer_quality_trends" in save_data:
                save_data["answer_quality_trends"] = save_data["answer_quality_trends"][-1000:]
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 (성능 최적화)"""
        start_time = datetime.now()
        
        # 캐시 확인
        question_hash = hashlib.md5(question.encode()).hexdigest()
        if question_hash in self.domain_cache:
            self.analysis_history["pattern_cache_stats"]["hits"] += 1
            return self.domain_cache[question_hash]
        
        self.analysis_history["pattern_cache_stats"]["misses"] += 1
        
        question_lower = question.lower()
        
        # 도메인 찾기 (성능 최적화)
        detected_domains = []
        domain_scores = {}
        
        # Trie를 사용한 빠른 키워드 매칭
        for domain in self.domain_keywords.keys():
            found_keywords = self._search_keywords_fast(question_lower, domain)
            if found_keywords:
                score = 0
                for keyword in found_keywords:
                    # 중요 키워드 가중치 적용
                    if keyword in ["개인정보보호법", "전자금융거래법", "자본시장법", "정보통신망법"]:
                        score += 3
                    elif len(keyword) >= 4:
                        score += 2
                    else:
                        score += 1
                
                domain_scores[domain] = score
        
        # 가장 높은 점수의 도메인들 선택
        if domain_scores:
            max_score = max(domain_scores.values())
            detected_domains = [domain for domain, score in domain_scores.items() 
                              if score >= max_score * 0.7]
        
        if not detected_domains:
            detected_domains = ["일반"]
        
        # 복잡도 계산 (최적화)
        complexity = self._calculate_enhanced_complexity(question)
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 대회 규칙 준수 확인
        compliance_check = self._check_enhanced_compliance(question)
        
        # 기관 관련 질문인지 확인
        institution_info = self._check_enhanced_institution_question(question)
        
        # 질문 의도 분석
        intent_analysis = self._analyze_enhanced_question_intent(question)
        
        # 전문성 수준 평가
        expertise_level = self._evaluate_expertise_level(question, korean_terms, complexity)
        
        # 분석 결과
        analysis_result = {
            "domain": detected_domains,
            "primary_domain": detected_domains[0] if detected_domains else "일반",
            "domain_scores": domain_scores,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info,
            "intent_analysis": intent_analysis,
            "expertise_level": expertise_level,
            "question_quality": self._assess_question_quality(question),
            "recommended_answer_type": self._recommend_answer_type(intent_analysis, complexity),
            "found_keywords": self._get_all_found_keywords(question_lower)
        }
        
        # 캐시에 저장
        self.domain_cache[question_hash] = analysis_result
        
        # 이력에 추가
        self._add_to_enhanced_analysis_history(question, analysis_result)
        
        # 성능 메트릭 업데이트
        processing_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics["avg_response_time"] = \
            (self.performance_metrics["avg_response_time"] * 0.9 + processing_time * 0.1)
        
        return analysis_result
    
    def _get_all_found_keywords(self, question_lower: str) -> Dict[str, List[str]]:
        """모든 도메인에서 발견된 키워드 반환"""
        found_keywords = {}
        for domain in self.domain_keywords.keys():
            keywords = self._search_keywords_fast(question_lower, domain)
            if keywords:
                found_keywords[domain] = keywords
        return found_keywords
    
    def _calculate_enhanced_complexity(self, question: str) -> float:
        """복잡도 계산 (최적화)"""
        # 기본 길이 요소
        length_factor = min(len(question) / 300, 1.0)
        
        # 전문 용어 밀도
        technical_terms = self._find_korean_technical_terms(question)
        term_density = len(technical_terms) / max(len(question.split()), 1) * 10
        term_factor = min(term_density, 1.0)
        
        # 문장 구조 복잡도
        sentence_count = len([s for s in question.split('.') if s.strip()])
        structure_factor = min(sentence_count / 5, 1.0)
        
        # 도메인 교차도
        domain_count = sum(1 for domain_keywords in self.domain_keywords.values() 
                          if any(keyword in question.lower() for keyword in domain_keywords))
        domain_factor = min(domain_count / 3, 1.0)
        
        # 법령 참조도
        law_keywords = ["법", "법령", "법률", "규정", "조항", "기준", "원칙"]
        law_count = sum(1 for keyword in law_keywords if keyword in question)
        law_factor = min(law_count / 3, 1.0)
        
        # 가중 평균
        complexity = (length_factor * 0.2 + term_factor * 0.3 + structure_factor * 0.2 + 
                     domain_factor * 0.2 + law_factor * 0.1)
        
        return min(complexity, 1.0)
    
    def _find_korean_technical_terms(self, question: str) -> List[str]:
        """한국어 전문 용어 찾기 (최적화)"""
        found_terms = []
        question_lower = question.lower()
        
        # 모든 도메인의 키워드에서 검색
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    found_terms.append(keyword)
        
        return list(set(found_terms))  # 중복 제거
    
    def _check_enhanced_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인 (최적화)"""
        compliance = {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True,
            "professional_level": True,
            "law_compliance": True
        }
        
        # 한국어 비율 확인
        korean_chars = len([c for c in question if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
        total_chars = len([c for c in question if c.isalpha()])
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            compliance["korean_content"] = korean_ratio > 0.8
        
        # 도메인 적절성 확인
        found_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                found_domains.append(domain)
        
        compliance["appropriate_domain"] = len(found_domains) > 0
        
        # 전문성 수준 확인
        professional_indicators = [
            "보안", "위험", "관리", "정책", "체계", "절차", "조치", "기준",
            "법령", "규정", "분석", "평가", "진단", "모니터링"
        ]
        prof_count = sum(1 for indicator in professional_indicators if indicator in question)
        compliance["professional_level"] = prof_count > 0
        
        # 법령 준수성 확인
        legal_terms = ["법", "규정", "기준", "원칙", "지침", "가이드라인"]
        compliance["law_compliance"] = any(term in question for term in legal_terms)
        
        return compliance
    
    def _check_enhanced_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인 (최적화)"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_category": "general",
            "expected_answer_elements": []
        }
        
        # 기관을 묻는 질문인지 확인
        institution_patterns = [
            r'기관.*기술하세요', r'기관.*설명하세요', r'기관.*서술하세요',
            r'어떤.*기관', r'어느.*기관', r'기관.*어디', r'기관.*무엇',
            r'조정.*신청.*기관', r'분쟁.*조정.*기관', r'분쟁.*해결.*기관',
            r'신청.*수.*있는.*기관', r'조정.*담당.*기관',
            r'감독.*기관', r'관리.*기관', r'담당.*기관', r'주관.*기관', r'소관.*기관',
            r'신고.*기관', r'접수.*기관', r'상담.*기관', r'문의.*기관',
            r'위원회.*무엇', r'위원회.*어디', r'위원회.*설명', r'위원회.*역할'
        ]
        
        pattern_matches = sum(1 for pattern in institution_patterns 
                             if re.search(pattern, question_lower))
        
        if pattern_matches > 0:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 2, 1.0)
            institution_info["question_category"] = "institution_inquiry"
            
            # 분야별 기관 확인
            if any(word in question_lower for word in ["전자금융", "전자금융거래", "전자서명"]) and "분쟁" in question_lower:
                institution_info["institution_type"] = "전자금융분쟁조정"
                institution_info["expected_answer_elements"] = ["기관명", "소속", "역할", "근거법", "신청방법"]
                
            elif any(word in question_lower for word in ["개인정보", "정보주체", "개인정보보호"]):
                institution_info["institution_type"] = "개인정보보호"
                institution_info["expected_answer_elements"] = ["기관명", "역할", "신고기관", "근거법"]
        
        return institution_info
    
    def _analyze_enhanced_question_intent(self, question: str) -> Dict:
        """질문 의도 분석 (최적화)"""
        # 캐시 확인
        intent_hash = hashlib.md5(f"intent_{question}".encode()).hexdigest()[:8]
        if intent_hash in self.pattern_cache:
            return self.pattern_cache[intent_hash]
        
        intent_patterns = {
            "기관_묻기": [
                r'기관.*기술하세요', r'기관.*설명하세요', r'어떤.*기관', r'어느.*기관',
                r'조정.*기관', r'분쟁.*기관', r'담당.*기관', r'위원회.*무엇'
            ],
            "특징_묻기": [
                r'특징.*설명하세요', r'특성.*설명하세요', r'어떤.*특징', r'주요.*특징',
                r'성격.*설명', r'속성.*설명', r'특성.*무엇'
            ],
            "지표_묻기": [
                r'지표.*설명하세요', r'탐지.*지표', r'주요.*지표', r'어떤.*지표',
                r'징후.*설명', r'신호.*설명', r'패턴.*설명'
            ],
            "방안_묻기": [
                r'방안.*기술하세요', r'방안.*설명하세요', r'대응.*방안', r'해결.*방안',
                r'어떤.*방안', r'대책.*설명', r'조치.*방안'
            ],
            "절차_묻기": [
                r'절차.*설명하세요', r'어떤.*절차', r'처리.*절차', r'진행.*절차',
                r'과정.*설명', r'단계.*설명', r'순서.*설명'
            ]
        }
        
        intent_scores = {}
        for intent_type, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, question, re.IGNORECASE))
            if score > 0:
                intent_scores[intent_type] = score
        
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = intent_scores[primary_intent] / 3.0
        else:
            primary_intent = "일반"
            confidence = 0.5
        
        result = {
            "primary_intent": primary_intent,
            "confidence": min(confidence, 1.0),
            "intent_scores": intent_scores,
            "secondary_intents": [intent for intent in intent_scores.keys() if intent != primary_intent]
        }
        
        # 캐시에 저장
        self.pattern_cache[intent_hash] = result
        
        return result
    
    def _evaluate_expertise_level(self, question: str, korean_terms: List[str], complexity: float) -> str:
        """전문성 수준 평가"""
        expertise_score = 0
        
        # 전문 용어 밀도
        if len(korean_terms) >= 3:
            expertise_score += 0.4
        elif len(korean_terms) >= 1:
            expertise_score += 0.2
        
        # 복잡도 기여
        expertise_score += complexity * 0.3
        
        # 법령 참조도
        law_terms = ["법", "법령", "규정", "조항", "기준"]
        law_count = sum(1 for term in law_terms if term in question)
        if law_count >= 2:
            expertise_score += 0.3
        elif law_count >= 1:
            expertise_score += 0.1
        
        # 수준 결정
        if expertise_score >= 0.7:
            return "전문가"
        elif expertise_score >= 0.4:
            return "중급"
        else:
            return "초급"
    
    def _assess_question_quality(self, question: str) -> Dict:
        """질문 품질 평가"""
        quality_score = 0
        quality_factors = {}
        
        # 길이 적절성
        length = len(question)
        if 50 <= length <= 300:
            quality_factors["length"] = 1.0
            quality_score += 0.2
        elif 30 <= length < 50 or 300 < length <= 400:
            quality_factors["length"] = 0.7
            quality_score += 0.1
        else:
            quality_factors["length"] = 0.3
        
        # 명확성
        question_marks = question.count('?')
        clear_indicators = ["설명하세요", "기술하세요", "무엇", "어떤", "어떻게"]
        clarity_score = min((question_marks + sum(1 for ind in clear_indicators if ind in question)) / 2, 1.0)
        quality_factors["clarity"] = clarity_score
        quality_score += clarity_score * 0.3
        
        # 전문성
        professional_terms = len(self._find_korean_technical_terms(question))
        prof_score = min(professional_terms / 2, 1.0)
        quality_factors["professionalism"] = prof_score
        quality_score += prof_score * 0.3
        
        # 구조적 완성도
        has_subject = any(word in question for word in ["은", "는", "이", "가"])
        has_predicate = any(word in question for word in ["하세요", "무엇", "어떤"])
        structure_score = (has_subject + has_predicate) / 2
        quality_factors["structure"] = structure_score
        quality_score += structure_score * 0.2
        
        return {
            "overall_score": min(quality_score, 1.0),
            "factors": quality_factors,
            "grade": "우수" if quality_score >= 0.8 else "양호" if quality_score >= 0.6 else "보통"
        }
    
    def _recommend_answer_type(self, intent_analysis: Dict, complexity: float) -> str:
        """답변 유형 추천"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        confidence = intent_analysis.get("confidence", 0.5)
        
        if confidence >= 0.7:
            if "기관" in primary_intent:
                return "기관명_구체적"
            elif "특징" in primary_intent:
                return "특징나열_상세"
            elif "지표" in primary_intent:
                return "지표목록_실무중심"
            elif "방안" in primary_intent:
                return "방안제시_단계별"
            elif "절차" in primary_intent:
                return "절차설명_순서별"
        
        if complexity >= 0.7:
            return "전문적_종합답변"
        elif complexity >= 0.4:
            return "표준_설명답변"
        else:
            return "기본_개념답변"
    
    def _determine_technical_level(self, complexity: float, korean_terms: List[str]) -> str:
        """기술적 수준 결정"""
        term_count = len(korean_terms)
        
        if complexity >= 0.7 and term_count >= 3:
            return "고급"
        elif complexity >= 0.4 and term_count >= 1:
            return "중급"
        else:
            return "기초"
    
    def _add_to_enhanced_analysis_history(self, question: str, analysis_result: Dict):
        """분석 이력 추가 (최적화)"""
        # 도메인 빈도 업데이트
        primary_domain = analysis_result["primary_domain"]
        self.analysis_history["domain_frequency"][primary_domain] += 1
        
        # 질문 패턴 추가 (최근 1000개만 유지)
        pattern_info = {
            "question_hash": hashlib.md5(question.encode()).hexdigest()[:8],
            "domain": primary_domain,
            "complexity": analysis_result["complexity"],
            "intent": analysis_result["intent_analysis"]["primary_intent"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history["question_patterns"].append(pattern_info)
        if len(self.analysis_history["question_patterns"]) > 1000:
            self.analysis_history["question_patterns"] = self.analysis_history["question_patterns"][-1000:]
        
        # 의도별 분석 이력
        intent = analysis_result["intent_analysis"]["primary_intent"]
        self.analysis_history["intent_analysis_history"][intent].append({
            "confidence": analysis_result["intent_analysis"]["confidence"],
            "complexity": analysis_result["complexity"],
            "quality": analysis_result["question_quality"]["overall_score"]
        })
        
        # 최근 50개만 유지
        if len(self.analysis_history["intent_analysis_history"][intent]) > 50:
            self.analysis_history["intent_analysis_history"][intent] = \
                self.analysis_history["intent_analysis_history"][intent][-50:]
    
    def get_korean_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """한국어 주관식 답변 템플릿 반환 (성능 최적화)"""
        start_time = datetime.now()
        
        # 템플릿 사용 통계 업데이트
        template_key = f"{domain}_{intent_type}"
        self.analysis_history["template_usage_stats"][template_key] += 1
        
        # 고품질 템플릿 우선 선택
        high_quality_template = self._get_high_quality_template(domain, intent_type)
        if high_quality_template:
            self.performance_metrics["template_selection_time"] = \
                (datetime.now() - start_time).total_seconds()
            return high_quality_template
        
        # 도메인과 의도에 맞는 템플릿 선택
        if domain in self.korean_subjective_templates:
            domain_templates = self.korean_subjective_templates[domain]
            
            if isinstance(domain_templates, dict):
                if intent_type in domain_templates:
                    templates = domain_templates[intent_type]
                elif "일반" in domain_templates:
                    templates = domain_templates["일반"]
                else:
                    templates = list(domain_templates.values())[0]
            else:
                templates = domain_templates
        else:
            templates = self.korean_subjective_templates["일반"]["일반"]
        
        # 품질 기반 템플릿 선택
        if isinstance(templates, list) and len(templates) > 1:
            quality_scores = []
            for template in templates:
                quality = self._evaluate_template_quality(template, intent_type)
                quality_scores.append((template, quality))
            
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            top_templates = [t for t, q in quality_scores[:min(3, len(quality_scores))]]
            selected_template = random.choice(top_templates)
        else:
            selected_template = random.choice(templates) if isinstance(templates, list) else templates
        
        # 한국어 전용 검증
        selected_template = self._ensure_korean_only(selected_template)
        
        # 템플릿 효과성 기록
        self._record_template_effectiveness(template_key, selected_template)
        
        self.performance_metrics["template_selection_time"] = \
            (datetime.now() - start_time).total_seconds()
        
        return selected_template
    
    def _get_high_quality_template(self, domain: str, intent_type: str) -> str:
        """고품질 템플릿 은행에서 선택"""
        bank_key = f"{domain}_{intent_type}"
        
        if bank_key in self.analysis_history["high_quality_template_bank"]:
            templates = self.analysis_history["high_quality_template_bank"][bank_key]
            if templates:
                best_template = max(templates, key=lambda x: x.get("quality_score", 0))
                if best_template["quality_score"] >= 0.85:
                    best_template["usage_count"] += 1
                    return best_template["content"]
        
        return None
    
    def _evaluate_template_quality(self, template: str, intent_type: str) -> float:
        """템플릿 품질 평가"""
        score = 0.0
        
        # 길이 적절성 (20%)
        length = len(template)
        min_len, max_len = self.template_quality_criteria["length_range"]
        if min_len <= length <= max_len:
            score += 0.20
        elif length < min_len:
            score += (length / min_len) * 0.20
        else:
            score += (max_len / length) * 0.20
        
        # 한국어 비율 (20%)
        korean_chars = len(re.findall(r'[가-힣]', template))
        total_chars = len(re.sub(r'[^\w가-힣]', '', template))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        if korean_ratio >= self.template_quality_criteria["korean_ratio_min"]:
            score += 0.20
        else:
            score += korean_ratio * 0.20
        
        # 구조적 키워드 포함 (20%)
        structure_keywords = self.template_quality_criteria["structure_keywords"]
        found_structure = sum(1 for keyword in structure_keywords if keyword in template)
        score += min(found_structure / len(structure_keywords), 1.0) * 0.20
        
        # 의도별 키워드 포함 (20%)
        if intent_type in self.template_quality_criteria["intent_keywords"]:
            intent_keywords = self.template_quality_criteria["intent_keywords"][intent_type]
            found_intent = sum(1 for keyword in intent_keywords if keyword in template)
            score += min(found_intent / len(intent_keywords), 1.0) * 0.20
        else:
            score += 0.10
        
        # 전문용어 포함 (10%)
        professional_terms = self.template_quality_criteria["professional_terms"]
        found_prof = sum(1 for term in professional_terms if term in template)
        score += min(found_prof / len(professional_terms), 1.0) * 0.10
        
        # 품질 지표 포함 (10%)
        quality_indicators = self.template_quality_criteria["quality_indicators"]
        found_quality = sum(1 for indicator in quality_indicators if indicator in template)
        score += min(found_quality / len(quality_indicators), 1.0) * 0.10
        
        return min(score, 1.0)
    
    def _ensure_korean_only(self, template: str) -> str:
        """한국어 전용 확인 및 수정"""
        # 영어 단어 제거
        template = re.sub(r'[a-zA-Z]{2,}', '', template)
        
        # 특수문자 정리
        template = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', template)
        
        # 공백 정리
        template = re.sub(r'\s+', ' ', template).strip()
        
        return template
    
    def _record_template_effectiveness(self, template_key: str, template: str):
        """템플릿 효과성 기록"""
        effectiveness_record = {
            "template": template[:100],  # 처음 100자만 저장
            "usage_time": datetime.now().isoformat(),
            "quality_score": self._evaluate_template_quality(template, template_key.split('_')[-1])
        }
        
        self.analysis_history["template_effectiveness"][template_key].append(effectiveness_record)
        
        # 최근 20개만 유지
        if len(self.analysis_history["template_effectiveness"][template_key]) > 20:
            self.analysis_history["template_effectiveness"][template_key] = \
                self.analysis_history["template_effectiveness"][template_key][-20:]
    
    def get_performance_metrics(self) -> Dict:
        """성능 메트릭 반환"""
        cache_total = (self.analysis_history["pattern_cache_stats"]["hits"] + 
                      self.analysis_history["pattern_cache_stats"]["misses"])
        
        if cache_total > 0:
            self.performance_metrics["cache_hit_rate"] = \
                self.analysis_history["pattern_cache_stats"]["hits"] / cache_total
        
        return dict(self.performance_metrics)
    
    def optimize_cache(self):
        """캐시 최적화"""
        # 캐시 크기 제한
        max_cache_size = 1000
        
        if len(self.domain_cache) > max_cache_size:
            # LRU 방식으로 오래된 항목 제거
            sorted_items = sorted(self.domain_cache.items(), 
                                key=lambda x: x[1].get("timestamp", ""), reverse=True)
            self.domain_cache = dict(sorted_items[:max_cache_size])
        
        if len(self.pattern_cache) > max_cache_size:
            # 절반 크기로 축소
            items = list(self.pattern_cache.items())
            self.pattern_cache = dict(items[:max_cache_size//2])
    
    def get_domain_statistics(self) -> Dict:
        """도메인 통계 반환"""
        total_questions = sum(self.analysis_history["domain_frequency"].values())
        
        if total_questions == 0:
            return {}
        
        domain_stats = {}
        for domain, count in self.analysis_history["domain_frequency"].items():
            domain_stats[domain] = {
                "count": count,
                "percentage": (count / total_questions) * 100,
                "expertise_score": self.analysis_history["domain_expertise_score"].get(domain, 0.0)
            }
        
        return domain_stats
    
    def cleanup(self):
        """정리 작업"""
        # 분석 이력 저장
        self._save_analysis_history()
        
        # 캐시 최적화
        self.optimize_cache()
        
        # 메모리 정리
        import gc
        gc.collect()