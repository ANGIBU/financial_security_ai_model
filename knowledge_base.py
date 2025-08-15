# knowledge_base.py

"""
금융보안 지식베이스 (최종 성능 강화)
- 도메인별 키워드 분류
- 전문 용어 처리
- 한국어 전용 답변 템플릿 제공 (품질 개선)
- 대회 규칙 준수 검증
- 질문 의도별 지식 제공 강화
- 고품질 템플릿 관리 시스템
"""

import pickle
import os
import re
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import random

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스 (최종 강화)"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 도메인별 키워드 (2025년 8월 1일 이전 공개 정보 기준) - 대폭 확장
        self.domain_keywords = {
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
                "개인정보유출", "개인정보오남용", "개인정보도용"
            ],
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
                "전자금융업신고", "전자금융업등록", "전자금융업인가"
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
        
        # 한국어 전용 주관식 답변 템플릿 (대폭 강화) - 의도별 확장
        self.korean_subjective_templates = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격접근도구는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다. 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다. 탐지를 회피하기 위한 다양한 기법을 사용하며 지속적인 위협을 가합니다.",
                    "트로이 목마는 유익한 프로그램으로 가장하여 사용자가 직접 설치하도록 유도하고, 설치 후 악의적인 기능을 수행하는 특징을 가집니다. 원격 접근 기능을 통해 시스템을 외부에서 조작할 수 있으며, 백도어 생성과 정보 유출이 주요 특징입니다.",
                    "원격접근 도구의 주요 특징은 은밀한 설치, 지속적인 연결 유지, 시스템 전반에 대한 제어권 획득, 사용자 모르게 정보 수집 등이며, 탐지를 회피하기 위한 다양한 기법을 사용합니다. 자동 실행과 은닉 기능을 통해 장기간 시스템에 잠복합니다.",
                    "악성 원격접근 도구는 정상 소프트웨어로 위장하여 배포되며, 설치 후 시스템 권한을 탈취하고 외부 서버와 은밀한 통신을 수행하는 특성을 가집니다. 루트킷 기능과 안티디버깅 기법을 통해 분석과 탐지를 어렵게 만듭니다."
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다. 시스템 성능 저하와 예상치 못한 네트워크 활동도 중요한 지표가 됩니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다. 로그 분석과 행위 기반 탐지를 통한 이상 징후 식별이 중요합니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다. DNS 쿼리 패턴과 통신 프로토콜 분석도 중요한 지표입니다.",
                    "비정상적인 아웃바운드 연결, 시스템 리소스 과다 사용, 백그라운드 프로세스 증가, 보안 소프트웨어 비활성화 시도, 시스템 설정 변경 등의 징후를 종합적으로 분석해야 합니다. 베이스라인 대비 편차 분석이 핵심입니다.",
                    "네트워크 연결 로그 분석, 프로세스 모니터링, 파일 무결성 검사, 레지스트리 변경 감시, 시스템 콜 추적 등을 통해 악성 활동을 탐지할 수 있습니다. 통계적 이상 탐지와 머신러닝 기반 분석을 활용한 지능형 탐지가 효과적입니다."
                ],
                "방안_묻기": [
                    "다층 방어체계 구축을 통한 예방, 실시간 모니터링 시스템 운영, 침입탐지 및 차단 시스템 도입, 정기적인 보안교육 실시, 보안 패치 관리 체계 운영 등의 종합적 대응방안이 필요합니다. 제로트러스트 보안 모델 적용과 엔드포인트 보안 강화가 중요합니다.",
                    "네트워크 분할을 통한 격리, 접근권한 최소화 원칙 적용, 행위 기반 탐지 시스템 구축, 사고 대응 절차 수립, 백업 및 복구 체계 마련 등의 보안 강화 방안을 수립해야 합니다. 위협 인텔리전스 활용과 보안 오케스트레이션 도입이 효과적입니다.",
                    "엔드포인트 보안 강화, 네트워크 트래픽 모니터링, 사용자 인식 개선 교육, 보안 정책 수립 및 준수, 정기적인 보안 점검 등을 통해 종합적인 보안 관리체계를 구축해야 합니다. AI 기반 위협 탐지와 자동화된 대응 체계 구축이 필요합니다.",
                    "보안 인식 교육 강화, 이메일 보안 솔루션 도입, 웹 필터링 시스템 구축, 정기적인 보안 진단 실시, 침해지표 공유 체계 구축 등의 예방적 보안 조치를 시행해야 합니다. 클라우드 보안과 모바일 보안 강화도 필수적입니다.",
                    "사이버 위협 인텔리전스 활용, 보안 운영 센터 구축, 자동화된 보안 분석 도구 도입, 지속적인 보안 모니터링, 신속한 사고 대응 체계 구축 등을 통해 고도화된 보안 체계를 운영해야 합니다."
                ],
                "일반": [
                    "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다. 보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다.",
                    "보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다. 위협 인텔리전스를 활용한 선제적 대응과 보안 인시던트 대응 계획 수립이 필요합니다.",
                    "악성코드 탐지를 위한 행위 기반 분석과 시그니처 기반 탐지를 병행하고, 네트워크 트래픽 모니터링을 통해 이상 징후를 조기에 발견해야 합니다. 제로데이 공격 대응과 APT 공격 탐지 역량 강화가 중요합니다.",
                    "사이버 보안 체계는 예방, 탐지, 대응, 복구의 단계별 접근을 통해 구축되어야 하며, 각 단계별로 적절한 보안 통제와 기술적 대응 방안을 마련해야 합니다. 보안 거버넌스와 컴플라이언스 준수가 필수적입니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보 침해신고센터에서 신고 접수 및 상담 업무를 담당합니다. 개인정보 관련 분쟁의 조정은 개인정보보호위원회 내 개인정보 분쟁조정위원회에서 담당하며, 피해구제와 분쟁해결 업무를 수행합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다. 개인정보 처리자에 대한 조사와 제재 권한도 가지고 있습니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있습니다. 온라인과 오프라인을 통한 다양한 신고 접수 채널을 운영하며 24시간 신고 접수가 가능합니다.",
                    "개인정보 관련 분쟁의 조정은 개인정보보호위원회 내 개인정보 분쟁조정위원회에서 담당하며, 피해구제와 분쟁해결 업무를 수행합니다. 집단분쟁조정과 개별분쟁조정을 통해 신속하고 공정한 해결을 도모합니다.",
                    "개인정보보호 정책 수립과 법령 집행은 개인정보보호위원회에서 담당하고, 침해신고 접수와 상담은 개인정보침해신고센터에서 처리합니다. 피해구제 신청과 집단분쟁조정은 개인정보 분쟁조정위원회에서 담당합니다."
                ],
                "방안_묻기": [
                    "개인정보 처리 시 수집 최소화 원칙 적용, 목적 외 이용 금지, 적절한 보호조치 수립, 정기적인 개인정보 영향평가 실시, 정보주체 권리 보장 체계 구축 등의 관리방안이 필요합니다. 개인정보 라이프사이클 전반에 걸친 보호조치 적용이 중요합니다.",
                    "개인정보보호 관리체계 구축, 개인정보처리방침 수립 및 공개, 개인정보보호책임자 지정, 정기적인 교육 실시, 기술적·관리적·물리적 보호조치 이행 등을 체계적으로 수행해야 합니다. 개인정보 처리 현황 관리와 정기적인 점검이 필수적입니다.",
                    "개인정보 수집 시 동의 절차 준수, 처리목적 명확화, 보유기간 설정 및 준수, 정보주체 권리 행사 절차 마련, 개인정보 파기 체계 구축 등의 전 과정 관리방안을 수립해야 합니다. 가명정보와 익명정보 처리 기준 마련도 중요합니다.",
                    "개인정보 처리 투명성 확보, 정보주체 통제권 강화, 개인정보 최소처리 원칙 준수, 처리목적 달성 시 즉시 파기, 개인정보 영향평가 정기 실시 등의 원칙을 철저히 준수해야 합니다. 개인정보 처리 시스템의 보안성 확보가 필수적입니다.",
                    "개인정보보호 거버넌스 체계 구축, 개인정보 처리자 및 취급자 교육 강화, 개인정보 침해 예방을 위한 기술적 조치 강화, 정보주체 권익 보호를 위한 절차 개선, 개인정보 보호수준 진단 및 개선 등을 지속적으로 추진해야 합니다."
                ],
                "일반": [
                    "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다. 개인정보 처리의 적법성과 정당성을 확보하고 정보주체의 동의를 받아야 합니다.",
                    "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다. 개인정보 침해 시 즉시 신고하고 피해 최소화 조치를 취해야 합니다.",
                    "개인정보 수집 시 수집목적과 이용범위를 명확히 고지하고 정보주체의 명시적 동의를 받아야 하며, 수집된 개인정보는 목적 달성 후 지체없이 파기해야 합니다. 개인정보 처리 방침을 수립하고 공개해야 합니다.",
                    "개인정보 처리는 적법하고 정당한 수단에 의해서만 가능하며, 처리목적에 필요한 최소한의 범위 내에서 이루어져야 합니다. 개인정보의 정확성과 최신성을 유지하고 안전성 확보를 위한 조치를 지속적으로 이행해야 합니다.",
                    "개인정보보호 원칙과 정보주체의 기본권 보장을 위해 수집 최소화, 목적 구속, 이용 제한, 정확성, 안전성, 투명성의 원칙을 준수하고 개인정보 라이프사이클 전반에 걸친 보호조치를 적용해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 전자금융거래법에 따라 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다. 전자금융거래에서 발생하는 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결합니다.",
                    "전자금융거래법에 따라 금융감독원의 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁의 조정 업무를 담당하고 있습니다. 이용자 보호와 금융시장 안정을 위한 분쟁해결 기구로 기능합니다.",
                    "전자금융 분쟁조정은 금융감독원에 설치된 전자금융분쟁조정위원회에서 신청할 수 있으며, 이용자 보호를 위한 분쟁해결 업무를 수행합니다. 무료로 분쟁조정 서비스를 제공하여 이용자의 권익을 보호합니다.",
                    "전자금융거래 분쟁의 조정은 금융감독원 전자금융분쟁조정위원회에서 담당하며, 공정하고 신속한 분쟁해결을 위한 업무를 수행합니다. 전자금융거래법에 근거하여 이용자와 전자금융업자 간의 분쟁을 조정합니다."
                ],
                "방안_묻기": [
                    "접근매체 보안 강화, 전자서명 및 인증체계 고도화, 거래내역 실시간 통지, 이상거래 탐지시스템 구축, 이용자 교육 강화 등의 종합적인 보안방안이 필요합니다. 생체인증과 다중인증 시스템 도입이 효과적입니다.",
                    "전자금융업자의 보안조치 의무 강화, 이용자 피해보상 체계 개선, 분쟁조정 절차 신속화, 보안기술 표준화, 관련 법령 정비 등의 제도적 개선방안을 추진해야 합니다. 핀테크 보안 가이드라인 수립과 준수가 중요합니다.",
                    "다중 인증 체계 도입, 거래한도 설정 및 관리, 보안카드 및 이용자 신원확인 강화, 금융사기 예방 시스템 구축, 이용자 보호 교육 확대 등을 실시해야 합니다. 모바일 금융 보안과 클라우드 보안 강화가 필요합니다.",
                    "전자금융 보안 표준 준수, 접근매체의 안전한 관리, 거래정보 암호화 전송, 이상거래 모니터링 시스템 구축, 보안사고 대응체계 마련 등의 기술적 보안조치를 강화해야 합니다. 오픈뱅킹 보안과 API 보안도 중요합니다.",
                    "전자금융 생태계 전반의 보안성 강화, 신기술 도입에 따른 보안 위험 관리, 국제 보안 표준 준수, 금융 사이버보안 협력체계 구축, 전자금융업 감독체계 고도화 등을 통한 종합적 보안체계 구축이 필요합니다."
                ],
                "일반": [
                    "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다. 이용자 보호와 건전한 전자금융시장 발전을 위한 제도적 기반을 제공합니다.",
                    "전자금융분쟁조정위원회에서 전자금융거래 분쟁조정 업무를 담당하며 이용자는 관련 법령에 따라 분쟁조정을 신청할 수 있습니다. 전자금융거래의 투명성과 신뢰성 확보가 중요합니다.",
                    "전자금융업자는 접근매체의 위조나 변조를 방지하기 위한 대책을 강구하고 이용자에게 안전한 거래환경을 제공해야 합니다. 전자금융거래 시 보안프로토콜 준수와 이용자 인증이 필수적입니다.",
                    "전자금융거래의 무결성과 기밀성 보장을 위해 강력한 암호화 기술 적용, 접근통제 시스템 구축, 거래로그 관리, 보안감사 실시 등의 종합적인 보안관리가 필요합니다. 디지털 금융 혁신과 보안의 균형이 중요합니다.",
                    "전자금융 서비스의 지속적인 발전과 이용자 편의성 증대를 위해 보안기술 혁신, 규제 샌드박스 활용, 업계 표준 수립, 국제 협력 강화 등의 다각적 접근이 필요합니다."
                ]
            },
            "정보보안": {
                "방안_묻기": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다. 보안 거버넌스 체계 구축과 최고경영진의 보안 의지가 중요합니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다. 제로트러스트 보안 모델 적용과 지속적인 보안성 검증이 필요합니다.",
                    "정보자산 분류체계를 구축하고 중요도에 따른 차등 보안조치를 적용하며, 정기적인 보안교육과 인식제고 프로그램을 운영해야 합니다. 클라우드 보안과 모바일 보안 강화도 필수적입니다.",
                    "물리적 보안조치, 기술적 보안조치, 관리적 보안조치를 균형있게 적용하고, 지속적인 보안성 평가와 개선활동을 수행해야 합니다. 인공지능과 빅데이터를 활용한 지능형 보안 시스템 구축이 효과적입니다.",
                    "보안 위험 관리 체계 수립, 보안 사고 예방 및 대응 절차 구축, 보안 기술 표준 준수, 보안 인력 전문성 강화, 보안 문화 조성 등의 종합적인 보안 관리 방안을 마련해야 합니다."
                ],
                "절차_묻기": [
                    "정보보안 관리절차는 보안정책 수립, 위험분석 실시, 보안대책 선정 및 구현, 보안교육 실시, 보안점검 및 감사, 보안사고 대응, 지속적 개선의 단계로 진행됩니다. PDCA 사이클에 따른 지속적인 관리가 중요합니다.",
                    "보안관리 절차는 계획 단계에서 정책과 기준을 수립하고, 구현 단계에서 보안조치를 적용하며, 운영 단계에서 모니터링과 관리를 수행하고, 개선 단계에서 평가와 보완을 실시합니다. 위험 기반 접근법 적용이 효과적입니다.",
                    "정보보안 업무 절차는 정보자산 식별 및 분류, 위험평가 실시, 보안대책 수립, 보안조치 이행, 보안수준 점검, 보안사고 처리, 보안성 개선의 순서로 수행됩니다. 국제 표준과 모범사례 적용이 중요합니다.",
                    "정보보안 관리체계 운영절차는 보안 거버넌스 구축, 보안 정책 및 절차 수립, 보안 조직 및 역할 정의, 보안 교육 및 인식 제고, 보안 모니터링 및 측정, 지속적 개선 활동의 단계로 구성됩니다.",
                    "ISMS 인증 기준에 따른 정보보안 관리절차는 관리체계 수립, 위험관리, 보안대책, 사후관리의 4단계로 구성되며, 각 단계별로 세부 통제항목을 이행하고 지속적인 개선을 수행해야 합니다."
                ],
                "일반": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다. 보안 거버넌스와 최고경영진의 의지가 성공의 핵심요소입니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다. 보안 위협의 진화에 대응한 동적 보안 체계 구축이 필요합니다.",
                    "정보자산의 기밀성, 무결성, 가용성을 보장하기 위한 종합적인 보안 체계를 구축하고, 내부자 위협과 외부 사이버 공격에 대응할 수 있는 다층 방어 체계를 운영해야 합니다.",
                    "정보보안 사고 예방을 위한 선제적 보안 조치와 사고 발생 시 신속한 대응을 위한 사고 대응 체계를 구축하며, 보안 인식 문화 조성과 지속적인 보안 역량 강화가 필요합니다."
                ]
            },
            "금융투자": {
                "방안_묻기": [
                    "투자자 보호를 위한 적합성 원칙 준수, 투자위험 고지 의무 이행, 투자자문 서비스 품질 개선, 불공정거래 방지 체계 구축, 내부통제 시스템 강화 등의 방안이 필요합니다. ESG 투자와 지속가능금융 확산도 중요합니다.",
                    "금융투자업자의 영업행위 규준 강화, 투자자 교육 확대, 분쟁조정 절차 개선, 시장감시 체계 고도화, 투자자 보호기금 운영 내실화 등을 추진해야 합니다. 디지털 자산과 가상자산 규제 체계 정비도 필요합니다.",
                    "투자상품 설명의무 강화, 투자자 유형별 맞춤형 서비스 제공, 투자권유 과정의 투명성 제고, 이해상충 방지 체계 구축, 투자자 피해구제 절차 개선 등이 필요합니다. 로보어드바이저와 핀테크 서비스 확산에 따른 규제 혁신이 중요합니다.",
                    "금융투자업 리스크 관리 체계 강화, 자본 건전성 규제 고도화, 시장 유동성 관리, 시스템 리스크 모니터링, 거시건전성 정책 운영 등의 시장 안정화 방안이 필요합니다.",
                    "금융투자시장의 글로벌 경쟁력 강화, 혁신금융 서비스 육성, 자본시장 인프라 고도화, 국제 규제 협력 강화, 금융 소비자 보호 체계 개선 등의 종합적 발전 방안을 추진해야 합니다."
                ],
                "일반": [
                    "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다. 투자자의 이익을 최우선으로 하는 수탁자 책임을 이행해야 합니다.",
                    "금융투자업 영위 시 투자자의 투자성향과 위험도를 평가하고 적합한 상품을 권유하는 적합성 원칙을 준수해야 합니다. 투자 설명서 제공과 투자위험 고지가 의무사항입니다.",
                    "투자자문업자는 고객의 투자목적과 재정상황을 종합적으로 고려하여 적절한 투자자문을 제공하고 이해상충을 방지해야 합니다. 투자자의 최선의 이익을 위한 선관주의 의무를 부담합니다.",
                    "금융투자상품의 복잡성과 위험도가 증가함에 따라 투자자 보호를 위한 규제 체계 고도화와 금융 소비자 권익 보호 강화가 필요합니다. 행동경제학 기반 투자자 보호 정책 개발이 중요합니다.",
                    "자본시장의 효율성과 투명성 제고를 위해 공시제도 개선, 시장조성 활성화, 거래소 경쟁력 강화, 해외 진출 지원 등의 시장 발전 정책이 필요합니다."
                ]
            },
            "위험관리": {
                "방안_묻기": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다. 통합위험관리 체계와 스트레스 테스트 정기 실시가 중요합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다. 신종 위험과 복합 위험에 대한 대응 체계 구축이 필요합니다.",
                    "위험관리 정책과 절차를 수립하고 위험한도를 설정하여 관리하며, 위험관리 조직과 책임체계를 명확히 정의해야 합니다. 위험 문화 조성과 위험 커뮤니케이션 강화가 중요합니다.",
                    "위험관리 문화 조성, 위험관리 교육 강화, 위험보고 체계 구축, 위험관리 성과평가 체계 도입, 외부 위험요인 모니터링 강화 등을 실시해야 합니다. 기후변화와 ESG 위험 관리도 필수적입니다.",
                    "디지털 전환에 따른 신종 위험 관리, 사이버 위험 대응 체계 강화, 운영위험 관리 고도화, 모델 위험 관리, 컨덕트 위험 관리 등의 현대적 위험관리 방안을 도입해야 합니다."
                ],
                "절차_묻기": [
                    "위험관리 절차는 위험식별 단계에서 잠재적 위험요소를 파악하고, 위험평가 단계에서 위험의 발생가능성과 영향도를 분석하며, 위험대응 단계에서 적절한 대응전략을 수립하고, 위험모니터링 단계에서 지속적으로 관리합니다.",
                    "위험관리 프로세스는 위험환경 분석, 위험요소 식별, 위험측정 및 평가, 위험대응 전략 수립, 위험통제 활동 실시, 위험모니터링 및 보고의 순서로 진행됩니다. 각 단계별로 체계적인 절차와 방법론을 적용해야 합니다.",
                    "통합위험관리 절차는 전사적 위험관리 정책 수립, 부문별 위험관리 계획 수립, 위험측정 및 평가 실시, 위험한도 설정 및 관리, 위험보고서 작성, 위험관리 성과 평가의 단계로 구성됩니다.",
                    "위험관리 운영절차는 위험 지배구조 구축, 위험관리 전략 수립, 위험식별 및 평가, 위험 대응 및 통제, 위험 모니터링 및 보고, 위험관리 검토 및 개선의 단계로 이루어지며, 지속적인 개선이 필요합니다.",
                    "금융기관의 위험관리 절차는 리스크 거버넌스 체계 구축, 위험선호도 설정, 위험측정 모델 개발, 스트레스 테스트 실시, 위험한도 관리, 위험보고 체계 운영의 순서로 진행됩니다."
                ],
                "일반": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다. 전사적 위험관리 관점에서 통합적 접근이 필요합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다. 위험 기반 내부통제와 3선 방어 체계 구축이 효과적입니다.",
                    "위험관리는 조직의 목표 달성을 위해 불확실성을 관리하는 체계적인 과정으로, 위험식별부터 모니터링까지 전 과정에 걸친 통합적 관리가 필요합니다. 위험과 수익의 균형을 통한 가치 창출이 중요합니다.",
                    "급변하는 경영환경에서 전통적 위험뿐만 아니라 신종 위험, 복합 위험, 시스템 위험 등에 대한 선제적 대응 체계 구축이 필요하며, 위험관리의 디지털화와 지능화가 요구됩니다."
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
        
        # 한국어 전용 금융 전문 용어 사전 (대회 규칙 준수) - 확장
        self.korean_financial_terms = {
            "정보보안관리체계": "조직의 정보자산을 보호하기 위한 종합적인 관리체계",
            "개인정보관리체계": "개인정보의 안전한 처리를 위한 체계적 관리방안",
            "원격접근": "네트워크를 통해 원격지에서 컴퓨터 시스템에 접근하는 방식",
            "지능형지속위협": "특정 목표를 대상으로 장기간에 걸쳐 수행되는 고도화된 사이버공격",
            "데이터유출방지": "조직 내부의 중요 데이터가 외부로 유출되는 것을 방지하는 보안기술",
            "모바일기기관리": "조직에서 사용하는 모바일 기기의 보안을 관리하는 솔루션",
            "보안정보이벤트관리": "보안 정보와 이벤트를 통합적으로 관리하고 분석하는 시스템",
            "비즈니스연속성계획": "재해나 위기상황 발생 시 업무 연속성을 보장하기 위한 계획",
            "재해복구계획": "정보시스템 장애 시 신속한 복구를 위한 절차와 방법",
            "전자금융분쟁조정위원회": "전자금융거래 관련 분쟁조정을 담당하는 기관",
            "개인정보보호위원회": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관",
            "트로이목마": "정상 프로그램으로 위장하여 악의적인 기능을 수행하는 악성코드",
            "원격접근도구": "네트워크를 통해 원격지 시스템을 제어할 수 있는 소프트웨어",
            "침입탐지시스템": "네트워크나 시스템의 비정상적인 활동을 탐지하는 보안 시스템",
            "침입방지시스템": "네트워크 침입을 실시간으로 차단하는 보안 시스템",
            "보안관제센터": "조직의 보안 상황을 24시간 모니터링하는 센터",
            "취약점진단": "시스템이나 네트워크의 보안 취약점을 찾아내는 활동",
            "모의해킹": "실제 해킹 기법을 사용하여 시스템의 보안성을 점검하는 활동",
            "디지털포렌식": "디지털 증거를 수집하고 분석하는 과학수사 기법",
            "제로트러스트": "모든 접근을 기본적으로 신뢰하지 않는 보안 모델",
            "인공지능보안": "AI 기술을 활용한 지능형 보안 솔루션",
            "블록체인보안": "블록체인 기술의 보안 특성과 취약점 관리",
            "클라우드보안": "클라우드 환경에서의 데이터와 서비스 보안",
            "IoT보안": "사물인터넷 기기와 네트워크의 보안 관리"
        }
        
        # 기관별 구체적 정보 (강화)
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "전자금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁조정",
                "근거법": "전자금융거래법",
                "신청방법": "금융감독원 홈페이지 또는 방문 신청",
                "상세정보": "전자금융거래에서 발생하는 분쟁의 공정하고 신속한 해결을 위해 설치된 기구로, 이용자와 전자금융업자 간의 분쟁조정 업무를 담당합니다. 무료 분쟁조정 서비스를 제공하며 조정 결과는 법적 구속력을 가집니다.",
                "관련기관": ["금융감독원", "한국은행", "금융위원회"],
                "업무범위": ["전자금융거래 분쟁조정", "피해구제", "조정안 제시", "합의권고"]
            },
            "개인정보보호": {
                "기관명": "개인정보보호위원회",
                "소속": "국무총리 소속",
                "역할": "개인정보보호 정책 수립 및 감시",
                "근거법": "개인정보보호법",
                "신고기관": "개인정보침해신고센터",
                "상세정보": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관으로, 개인정보 보호 정책 수립, 법령 집행, 감시 업무를 수행합니다. 개인정보 처리자에 대한 조사와 제재 권한을 가지고 있습니다.",
                "관련기관": ["개인정보침해신고센터", "개인정보 분쟁조정위원회", "방송통신위원회"],
                "업무범위": ["정책수립", "법령집행", "조사제재", "분쟁조정", "교육홍보"]
            },
            "금융투자분쟁조정": {
                "기관명": "금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "금융투자 관련 분쟁조정",
                "근거법": "자본시장법",
                "상세정보": "금융투자업과 관련된 분쟁의 조정을 담당하며, 투자자 보호와 분쟁의 공정한 해결을 위한 업무를 수행합니다. 전문성과 독립성을 바탕으로 한 조정 서비스를 제공합니다.",
                "관련기관": ["금융감독원", "금융위원회", "한국거래소"],
                "업무범위": ["투자분쟁조정", "집단분쟁조정", "피해구제", "화해권고"]
            },
            "사이버보안": {
                "기관명": "한국인터넷진흥원",
                "소속": "과학기술정보통신부",
                "역할": "사이버보안 정책 수립 및 대응",
                "근거법": "정보통신망법",
                "상세정보": "사이버보안 정책 수립과 대응체계 구축, 보안기술 개발, 보안인력 양성 등의 업무를 수행하며, 국가 사이버보안 컨트롤타워 역할을 담당합니다.",
                "관련기관": ["과학기술정보통신부", "국가정보원", "경찰청"],
                "업무범위": ["정책수립", "기술개발", "인력양성", "사고대응", "국제협력"]
            }
        }
        
        # 템플릿 품질 평가 기준 (신규)
        self.template_quality_criteria = {
            "length_range": (80, 450),  # 적절한 길이 범위 확장
            "korean_ratio_min": 0.95,   # 최소 한국어 비율 강화
            "structure_keywords": ["법", "규정", "조치", "관리", "절차", "기준", "체계", "정책"],  # 구조적 키워드 확장
            "intent_keywords": {
                "기관_묻기": ["위원회", "기관", "담당", "업무", "조정", "분쟁", "신고", "접수"],
                "특징_묻기": ["특징", "특성", "성질", "기능", "속성", "성격", "원리", "메커니즘"],
                "지표_묻기": ["지표", "징후", "패턴", "탐지", "신호", "증상", "단서", "흔적"],
                "방안_묻기": ["방안", "대책", "조치", "관리", "대응", "해결", "개선", "강화"],
                "절차_묻기": ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행", "처리"],
                "조치_묻기": ["조치", "대응", "대책", "보안", "예방", "개선", "강화", "보완"],
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
            "domain_frequency": {},
            "complexity_distribution": {},
            "question_patterns": [],
            "compliance_check": {
                "korean_only": 0,
                "law_references": 0,
                "technical_terms": 0
            },
            "intent_analysis_history": {},  # 의도 분석 이력
            "template_usage_stats": {},     # 템플릿 사용 통계
            "template_effectiveness": {},   # 템플릿 효과성
            "high_quality_template_bank": {},  # 고품질 템플릿 은행 (신규)
            "domain_expertise_score": {},  # 도메인별 전문성 점수
            "answer_quality_trends": [],   # 답변 품질 트렌드
            "best_practice_patterns": {}   # 모범사례 패턴
        }
        
        # 이전 분석 이력 로드
        self._load_analysis_history()
    
    def _load_analysis_history(self):
        """이전 분석 이력 로드"""
        history_file = self.pkl_dir / "analysis_history.pkl"
        
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    saved_history = pickle.load(f)
                    self.analysis_history.update(saved_history)
            except Exception:
                pass
    
    def _save_analysis_history(self):
        """분석 이력 저장"""
        history_file = self.pkl_dir / "analysis_history.pkl"
        
        try:
            save_data = {
                **self.analysis_history,
                "last_updated": datetime.now().isoformat()
            }
            
            # 최근 2000개 패턴만 저장 (확장)
            save_data["question_patterns"] = save_data["question_patterns"][-2000:]
            save_data["answer_quality_trends"] = save_data["answer_quality_trends"][-1000:]
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 (대회 규칙 준수 확인) - 강화"""
        question_lower = question.lower()
        
        # 도메인 찾기 (정밀도 향상)
        detected_domains = []
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    # 키워드 중요도에 따른 가중치 적용
                    if keyword in ["개인정보보호법", "전자금융거래법", "자본시장법", "정보통신망법"]:
                        score += 3  # 법령명은 높은 가중치
                    elif len(keyword) >= 4:
                        score += 2  # 긴 키워드는 높은 정확도
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        # 가장 높은 점수의 도메인들 선택
        if domain_scores:
            max_score = max(domain_scores.values())
            detected_domains = [domain for domain, score in domain_scores.items() 
                              if score >= max_score * 0.7]  # 70% 이상 점수만
        
        if not detected_domains:
            detected_domains = ["일반"]
        
        # 복잡도 계산 (개선)
        complexity = self._calculate_enhanced_complexity(question)
        
        # 한국어 전문 용어 포함 여부 (확장)
        korean_terms = self._find_korean_technical_terms(question)
        
        # 대회 규칙 준수 확인 (강화)
        compliance_check = self._check_enhanced_compliance(question)
        
        # 기관 관련 질문인지 확인 (강화)
        institution_info = self._check_enhanced_institution_question(question)
        
        # 질문 의도 분석 (정밀도 향상)
        intent_analysis = self._analyze_enhanced_question_intent(question)
        
        # 전문성 수준 평가 (신규)
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
            "recommended_answer_type": self._recommend_answer_type(intent_analysis, complexity)
        }
        
        # 이력에 추가 (강화)
        self._add_to_enhanced_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def _calculate_enhanced_complexity(self, question: str) -> float:
        """강화된 복잡도 계산"""
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
        domain_count = sum(1 for keywords in self.domain_keywords.values() 
                          if any(keyword in question.lower() for keyword in keywords))
        domain_factor = min(domain_count / 3, 1.0)
        
        # 법령 참조도
        law_keywords = ["법", "법령", "법률", "규정", "조항", "기준", "원칙"]
        law_count = sum(1 for keyword in law_keywords if keyword in question)
        law_factor = min(law_count / 3, 1.0)
        
        # 가중 평균
        complexity = (length_factor * 0.2 + term_factor * 0.3 + structure_factor * 0.2 + 
                     domain_factor * 0.2 + law_factor * 0.1)
        
        return min(complexity, 1.0)
    
    def _check_enhanced_compliance(self, question: str) -> Dict:
        """강화된 대회 규칙 준수 확인"""
        compliance = {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True,
            "professional_level": True,
            "law_compliance": True
        }
        
        # 한국어 비율 확인 (강화)
        korean_chars = len([c for c in question if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
        total_chars = len([c for c in question if c.isalpha()])
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            compliance["korean_content"] = korean_ratio > 0.8
        
        # 도메인 적절성 확인 (강화)
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
        """강화된 기관 관련 질문 확인"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_category": "general",
            "expected_answer_elements": []
        }
        
        # 기관을 묻는 질문인지 확인 (대폭 강화)
        institution_patterns = [
            # 직접적인 기관 질문
            r'기관.*기술하세요', r'기관.*설명하세요', r'기관.*서술하세요',
            r'어떤.*기관', r'어느.*기관', r'기관.*어디', r'기관.*무엇',
            
            # 조정/분쟁 관련
            r'조정.*신청.*기관', r'분쟁.*조정.*기관', r'분쟁.*해결.*기관',
            r'신청.*수.*있는.*기관', r'조정.*담당.*기관',
            
            # 감독/관리 기관
            r'감독.*기관', r'관리.*기관', r'담당.*기관', r'주관.*기관', r'소관.*기관',
            
            # 신고/접수 기관
            r'신고.*기관', r'접수.*기관', r'상담.*기관', r'문의.*기관',
            
            # 위원회 관련
            r'위원회.*무엇', r'위원회.*어디', r'위원회.*설명', r'위원회.*역할',
            
            # 업무 담당 기관
            r'업무.*담당.*기관', r'담당.*하는.*기관', r'관련.*기관.*어디',
            r'처리.*기관', r'운영.*기관', r'설치.*기관'
        ]
        
        pattern_matches = 0
        matched_patterns = []
        
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 1
                matched_patterns.append(pattern)
        
        is_asking_institution = pattern_matches > 0
        
        if is_asking_institution:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 2, 1.0)
            institution_info["question_category"] = "institution_inquiry"
            
            # 분야별 기관 확인 (정밀 매칭)
            if any(word in question_lower for word in ["전자금융", "전자금융거래", "전자서명"]) and "분쟁" in question_lower:
                institution_info["institution_type"] = "전자금융분쟁조정"
                institution_info["relevant_institution"] = self.institution_database["전자금융분쟁조정"]
                institution_info["expected_answer_elements"] = ["기관명", "소속", "역할", "근거법", "신청방법"]
                
            elif any(word in question_lower for word in ["개인정보", "정보주체", "개인정보보호"]):
                institution_info["institution_type"] = "개인정보보호"
                institution_info["relevant_institution"] = self.institution_database["개인정보보호"]
                institution_info["expected_answer_elements"] = ["기관명", "역할", "신고기관", "근거법"]
                
            elif any(word in question_lower for word in ["금융투자", "투자자문", "자본시장"]) and "분쟁" in question_lower:
                institution_info["institution_type"] = "금융투자분쟁조정"
                institution_info["relevant_institution"] = self.institution_database["금융투자분쟁조정"]
                institution_info["expected_answer_elements"] = ["기관명", "소속", "역할", "근거법"]
                
            elif any(word in question_lower for word in ["사이버보안", "사이버", "인터넷"]):
                institution_info["institution_type"] = "사이버보안"
                institution_info["relevant_institution"] = self.institution_database["사이버보안"]
                institution_info["expected_answer_elements"] = ["기관명", "소속", "역할", "업무범위"]
        
        return institution_info
    
    def _analyze_enhanced_question_intent(self, question: str) -> Dict:
        """강화된 질문 의도 분석"""
        # 기본 의도 분석 (기존 로직 활용)
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
            score = 0
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    score += 1
            if score > 0:
                intent_scores[intent_type] = score
        
        # 기본 의도 결정
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = intent_scores[primary_intent] / 3.0
        else:
            primary_intent = "일반"
            confidence = 0.5
        
        return {
            "primary_intent": primary_intent,
            "confidence": min(confidence, 1.0),
            "intent_scores": intent_scores,
            "secondary_intents": [intent for intent in intent_scores.keys() if intent != primary_intent]
        }
    
    def _evaluate_expertise_level(self, question: str, korean_terms: List[str], complexity: float) -> str:
        """전문성 수준 평가 (신규)"""
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
        """질문 품질 평가 (신규)"""
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
        """답변 유형 추천 (신규)"""
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
    
    def get_korean_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """한국어 주관식 답변 템플릿 반환 (품질 강화)"""
        
        # 템플릿 사용 통계 업데이트
        template_key = f"{domain}_{intent_type}"
        if template_key not in self.analysis_history["template_usage_stats"]:
            self.analysis_history["template_usage_stats"][template_key] = 0
        self.analysis_history["template_usage_stats"][template_key] += 1
        
        # 고품질 템플릿 우선 선택 (신규)
        high_quality_template = self._get_high_quality_template(domain, intent_type)
        if high_quality_template:
            return high_quality_template
        
        # 도메인과 의도에 맞는 템플릿 선택
        if domain in self.korean_subjective_templates:
            domain_templates = self.korean_subjective_templates[domain]
            
            # 의도별 템플릿이 있는지 확인
            if isinstance(domain_templates, dict):
                if intent_type in domain_templates:
                    templates = domain_templates[intent_type]
                elif "일반" in domain_templates:
                    templates = domain_templates["일반"]
                else:
                    # dict의 첫 번째 값 사용
                    templates = list(domain_templates.values())[0]
            else:
                templates = domain_templates
        else:
            # 일반 템플릿 사용
            templates = self.korean_subjective_templates["일반"]["일반"]
        
        # 품질 기반 템플릿 선택 (강화)
        if isinstance(templates, list) and len(templates) > 1:
            # 템플릿 품질 평가 후 선택
            quality_scores = []
            for template in templates:
                quality = self._evaluate_template_quality(template, intent_type)
                quality_scores.append((template, quality))
            
            # 상위 품질 템플릿 중에서 선택
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            top_templates = [t for t, q in quality_scores[:min(3, len(quality_scores))]]
            selected_template = random.choice(top_templates)
        else:
            selected_template = random.choice(templates) if isinstance(templates, list) else templates
        
        # 한국어 전용 검증 (대회 규칙 준수)
        selected_template = self._ensure_korean_only(selected_template)
        
        # 템플릿 효과성 기록 (강화)
        self._record_template_effectiveness(template_key, selected_template)
        
        return selected_template
    
    def _get_high_quality_template(self, domain: str, intent_type: str) -> str:
        """고품질 템플릿 은행에서 선택 (신규)"""
        bank_key = f"{domain}_{intent_type}"
        
        if bank_key in self.analysis_history["high_quality_template_bank"]:
            templates = self.analysis_history["high_quality_template_bank"][bank_key]
            if templates:
                # 품질 점수가 높은 템플릿 우선 선택
                best_template = max(templates, key=lambda x: x.get("quality_score", 0))
                if best_template["quality_score"] >= 0.85:
                    best_template["usage_count"] += 1
                    return best_template["content"]
        
        return None
    
    def _evaluate_template_quality(self, template: str, intent_type: str) -> float:
        """템플릿 품질 평가 (강화)"""
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
            score += 0.10  # 의도 키워드가 없는 경우 기본 점수
        
        # 전문용어 포함 (10%)
        professional_terms = self.template_quality_criteria["professional_terms"]
        found_prof = sum(1 for term in professional_terms if term in template)