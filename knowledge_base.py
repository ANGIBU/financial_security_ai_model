# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 분류
- 전문 용어 처리
- 한국어 전용 답변 템플릿 제공
- 대회 규칙 준수 검증
- 질문 의도별 지식 제공 강화
"""

import pickle
import os
import re
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import random

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 도메인별 키워드 (2025년 8월 1일 이전 공개 정보 기준) - 확장
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "수집", "이용", "제공", "파기", "동의",
                "법정대리인", "아동", "처리", "개인정보처리방침", "열람권",
                "정정삭제권", "처리정지권", "손해배상", "개인정보보호위원회",
                "개인정보영향평가", "개인정보관리체계", "개인정보처리시스템",
                "개인정보보호책임자", "개인정보취급자", "개인정보침해신고센터"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서", "전자금융업",
                "전자지급수단", "전자화폐", "전자금융거래", "인증",
                "전자금융분쟁조정위원회", "금융감독원", "한국은행",
                "전자금융거래기록", "전자금융업무", "전자적장치",
                "전자금융거래약관", "전자금융서비스", "전자금융업무위탁"
            ],
            "사이버보안": [
                "트로이", "악성코드", "해킹", "멀웨어", "피싱", 
                "스미싱", "랜섬웨어", "바이러스", "웜", "스파이웨어",
                "원격제어", "원격접근", "RAT", "봇넷", "분산서비스거부공격", 
                "지능형지속위협", "제로데이", "딥페이크", "사회공학", 
                "취약점", "패치", "침입탐지", "침입방지", "보안관제",
                "백도어", "루트킷", "키로거", "트로이목마", "원격접근도구"
            ],
            "정보보안": [
                "정보보안", "보안관리", "정보보안관리체계", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지시스템",
                "침입방지시스템", "보안정보이벤트관리", "보안관제", "인증",
                "권한관리", "로그관리", "백업", "복구", "재해복구",
                "비즈니스연속성계획", "보안감사", "보안교육", "ISMS"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "집합투자업", "신탁업", "소비자금융업", "보험중개업",
                "금융투자회사", "자본시장법", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "투자권유",
                "금융투자상품", "투자위험", "투자성과"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "위험회피",
                "위험전가", "위험감소", "위험분석", "위험식별", "위험모니터링",
                "리스크", "내부통제", "컴플라이언스", "감사", "위험통제",
                "위험보고", "위험문화", "위험거버넌스", "위험한도"
            ]
        }
        
        # 객관식 질문 패턴 (한국어 전용)
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
        
        # 한국어 전용 주관식 답변 템플릿 (대회 규칙 준수) - 의도별 확장
        self.korean_subjective_templates = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격접근도구는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "트로이 목마는 유용한 프로그램으로 가장하여 사용자가 직접 설치하도록 유도하고, 설치 후 악의적인 기능을 수행하는 특징을 가집니다. 원격 접근 기능을 통해 시스템을 외부에서 조작할 수 있습니다."
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다."
                ],
                "일반": [
                    "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다.",
                    "보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보 침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있습니다."
                ],
                "일반": [
                    "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다.",
                    "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영됩니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                    "전자금융거래법에 따라 금융감독원의 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁의 조정 업무를 담당하고 있습니다.",
                    "전자금융 분쟁조정은 금융감독원에 설치된 전자금융분쟁조정위원회에서 신청할 수 있으며, 이용자 보호를 위한 분쟁해결 업무를 수행합니다."
                ],
                "일반": [
                    "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다.",
                    "전자금융분쟁조정위원회에서 전자금융거래 분쟁조정 업무를 담당하며 이용자는 관련 법령에 따라 분쟁조정을 신청할 수 있습니다."
                ]
            },
            "정보보안": {
                "일반": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다."
                ]
            },
            "금융투자": {
                "일반": [
                    "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다.",
                    "금융투자업 영위 시 투자자의 투자성향과 위험도를 평가하고 적합한 상품을 권유하는 적합성 원칙을 준수해야 합니다."
                ]
            },
            "위험관리": {
                "일반": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다."
                ]
            },
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
                    "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다.",
                    "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다.",
                    "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다."
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
            "전자금융분쟁조정위원회": "전자금융거래 관련 분쟁의 조정을 담당하는 기관",
            "개인정보보호위원회": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관",
            "트로이목마": "정상 프로그램으로 위장하여 악의적 기능을 수행하는 악성코드",
            "원격접근도구": "네트워크를 통해 원격지 시스템을 제어할 수 있는 소프트웨어"
        }
        
        # 기관별 구체적 정보 (신규)
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "전자금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁조정",
                "근거법": "전자금융거래법",
                "신청방법": "금융감독원 홈페이지 또는 방문 신청"
            },
            "개인정보보호": {
                "기관명": "개인정보보호위원회",
                "소속": "국무총리 소속",
                "역할": "개인정보보호 정책 수립 및 감시",
                "근거법": "개인정보보호법",
                "신고기관": "개인정보침해신고센터"
            },
            "금융투자분쟁조정": {
                "기관명": "금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "금융투자 관련 분쟁조정",
                "근거법": "자본시장법"
            }
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
            "intent_analysis_history": {}  # 의도 분석 이력 추가
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
            
            # 최근 1000개 패턴만 저장 (대회 환경 고려)
            save_data["question_patterns"] = save_data["question_patterns"][-1000:]
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 (대회 규칙 준수 확인) - 강화"""
        question_lower = question.lower()
        
        # 도메인 찾기
        detected_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domains.append(domain)
        
        if not detected_domains:
            detected_domains = ["일반"]
        
        # 복잡도 계산
        complexity = self._calculate_complexity(question)
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 대회 규칙 준수 확인
        compliance_check = self._check_competition_compliance(question)
        
        # 기관 관련 질문인지 확인 (신규)
        institution_info = self._check_institution_question(question)
        
        # 분석 결과 저장
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info  # 기관 정보 추가
        }
        
        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인 (신규)"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None
        }
        
        # 기관을 묻는 질문인지 확인
        institution_patterns = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관"
        ]
        
        is_asking_institution = any(re.search(pattern, question_lower) for pattern in institution_patterns)
        
        if is_asking_institution:
            institution_info["is_institution_question"] = True
            
            # 분야별 기관 확인
            if "전자금융" in question_lower and "분쟁" in question_lower:
                institution_info["institution_type"] = "전자금융분쟁조정"
                institution_info["relevant_institution"] = self.institution_database["전자금융분쟁조정"]
            elif "개인정보" in question_lower:
                institution_info["institution_type"] = "개인정보보호"
                institution_info["relevant_institution"] = self.institution_database["개인정보보호"]
            elif "금융투자" in question_lower and "분쟁" in question_lower:
                institution_info["institution_type"] = "금융투자분쟁조정"
                institution_info["relevant_institution"] = self.institution_database["금융투자분쟁조정"]
        
        return institution_info
    
    def _check_competition_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인"""
        compliance = {
            "korean_content": True,  # 한국어 질문인지
            "appropriate_domain": True,  # 적절한 도메인인지
            "no_external_dependency": True  # 외부 의존성 없는지
        }
        
        # 한국어 비율 확인
        korean_chars = len([c for c in question if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
        total_chars = len([c for c in question if c.isalpha()])
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            compliance["korean_content"] = korean_ratio > 0.7
        
        # 도메인 적절성 확인
        found_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                found_domains.append(domain)
        
        compliance["appropriate_domain"] = len(found_domains) > 0
        
        return compliance
    
    def _add_to_analysis_history(self, question: str, analysis: Dict):
        """분석 이력에 추가"""
        # 도메인 빈도 업데이트
        for domain in analysis["domain"]:
            self.analysis_history["domain_frequency"][domain] = \
                self.analysis_history["domain_frequency"].get(domain, 0) + 1
        
        # 복잡도 분포 업데이트
        level = analysis["technical_level"]
        self.analysis_history["complexity_distribution"][level] = \
            self.analysis_history["complexity_distribution"].get(level, 0) + 1
        
        # 준수성 확인 업데이트
        if analysis["compliance"]["korean_content"]:
            self.analysis_history["compliance_check"]["korean_only"] += 1
        
        if any("법" in term for term in analysis["korean_technical_terms"]):
            self.analysis_history["compliance_check"]["law_references"] += 1
        
        if len(analysis["korean_technical_terms"]) > 0:
            self.analysis_history["compliance_check"]["technical_terms"] += 1
        
        # 기관 질문 이력 추가 (신규)
        if analysis["institution_info"]["is_institution_question"]:
            institution_type = analysis["institution_info"]["institution_type"]
            if institution_type not in self.analysis_history["intent_analysis_history"]:
                self.analysis_history["intent_analysis_history"][institution_type] = 0
            self.analysis_history["intent_analysis_history"][institution_type] += 1
        
        # 질문 패턴 추가
        pattern = {
            "question_length": len(question),
            "domain": analysis["domain"][0] if analysis["domain"] else "일반",
            "complexity": analysis["complexity"],
            "korean_terms_count": len(analysis["korean_technical_terms"]),
            "compliance_score": sum(analysis["compliance"].values()) / len(analysis["compliance"]),
            "is_institution_question": analysis["institution_info"]["is_institution_question"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history["question_patterns"].append(pattern)
    
    def get_korean_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """한국어 주관식 답변 템플릿 반환 (대회 규칙 준수) - 의도별 강화"""
        
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
        
        # 한국어 전용 검증
        selected_template = random.choice(templates)
        
        # 영어 문자 제거 (대회 규칙 준수)
        import re
        selected_template = re.sub(r'[a-zA-Z]+', '', selected_template)
        selected_template = re.sub(r'\s+', ' ', selected_template).strip()
        
        return selected_template
    
    def get_institution_specific_answer(self, institution_type: str) -> str:
        """기관별 구체적 답변 반환 (신규)"""
        if institution_type in self.institution_database:
            info = self.institution_database[institution_type]
            
            if institution_type == "전자금융분쟁조정":
                return f"{info['기관명']}에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 {info['소속']} 내에 설치되어 운영되며, {info['근거법']}에 따라 이용자의 분쟁조정 신청을 접수하고 처리합니다."
            
            elif institution_type == "개인정보보호":
                return f"{info['기관명']}이 개인정보 보호에 관한 업무를 총괄하며, {info['신고기관']}에서 신고 접수 및 상담 업무를 담당합니다. 이는 {info['근거법']}에 근거하여 운영됩니다."
            
            elif institution_type == "금융투자분쟁조정":
                return f"{info['기관명']}에서 금융투자 관련 분쟁조정 업무를 담당하며, {info['소속']} 내에 설치되어 {info['근거법']}에 따라 운영됩니다."
        
        # 기본 답변
        return "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다."
    
    def get_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """주관식 답변 템플릿 반환 (한국어 전용)"""
        return self.get_korean_subjective_template(domain, intent_type)
    
    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        # 길이 기반 복잡도
        length_factor = min(len(question) / 200, 1.0)
        
        # 한국어 전문 용어 개수
        korean_term_count = sum(1 for term in self.korean_financial_terms.keys() 
                               if term in question)
        term_factor = min(korean_term_count / 3, 1.0)
        
        # 도메인 개수
        domain_count = sum(1 for keywords in self.domain_keywords.values() 
                          if any(keyword in question.lower() for keyword in keywords))
        domain_factor = min(domain_count / 2, 1.0)
        
        return (length_factor + term_factor + domain_factor) / 3
    
    def _find_korean_technical_terms(self, question: str) -> List[str]:
        """한국어 전문 용어 찾기"""
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
    
    def get_domain_specific_guidance(self, domain: str) -> Dict:
        """도메인별 지침 반환 (대회 규칙 준수) - 강화"""
        guidance = {
            "개인정보보호": {
                "key_laws": ["개인정보보호법", "정보통신망법"],
                "key_concepts": ["정보주체", "개인정보처리자", "동의", "목적외이용금지"],
                "oversight_body": "개인정보보호위원회",
                "related_institutions": ["개인정보보호위원회", "개인정보침해신고센터"],
                "compliance_focus": "한국어 법령 용어 사용"
            },
            "전자금융": {
                "key_laws": ["전자금융거래법", "전자서명법"],
                "key_concepts": ["접근매체", "전자서명", "인증", "분쟁조정"],
                "oversight_body": "금융감독원, 한국은행",
                "related_institutions": ["전자금융분쟁조정위원회", "금융감독원"],
                "compliance_focus": "한국어 금융 용어 사용"
            },
            "사이버보안": {
                "key_laws": ["정보통신망법", "개인정보보호법"],
                "key_concepts": ["악성코드", "침입탐지", "보안관제", "사고대응"],
                "oversight_body": "과학기술정보통신부, 경찰청",
                "related_institutions": ["한국인터넷진흥원", "사이버보안센터"],
                "compliance_focus": "한국어 보안 용어 사용"
            },
            "정보보안": {
                "key_laws": ["정보통신망법", "전자정부법"],
                "key_concepts": ["정보보안관리체계", "접근통제", "암호화", "백업"],
                "oversight_body": "과학기술정보통신부",
                "related_institutions": ["한국인터넷진흥원"],
                "compliance_focus": "한국어 기술 용어 사용"
            },
            "금융투자": {
                "key_laws": ["자본시장법", "금융투자업규정"],
                "key_concepts": ["투자자보호", "적합성원칙", "설명의무", "내부통제"],
                "oversight_body": "금융감독원, 금융위원회",
                "related_institutions": ["금융분쟁조정위원회", "금융감독원"],
                "compliance_focus": "한국어 투자 용어 사용"
            },
            "위험관리": {
                "key_laws": ["은행법", "보험업법", "자본시장법"],
                "key_concepts": ["위험평가", "내부통제", "컴플라이언스", "감사"],
                "oversight_body": "금융감독원",
                "related_institutions": ["금융감독원"],
                "compliance_focus": "한국어 관리 용어 사용"
            }
        }
        
        return guidance.get(domain, {
            "key_laws": ["관련 법령"],
            "key_concepts": ["체계적 관리", "지속적 개선"],
            "oversight_body": "관계기관",
            "related_institutions": ["해당 전문기관"],
            "compliance_focus": "한국어 전용 답변"
        })
    
    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        return {
            "domain_frequency": dict(self.analysis_history["domain_frequency"]),
            "complexity_distribution": dict(self.analysis_history["complexity_distribution"]),
            "compliance_check": dict(self.analysis_history["compliance_check"]),
            "intent_analysis_history": dict(self.analysis_history["intent_analysis_history"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database)
        }
    
    def validate_competition_compliance(self, answer: str, domain: str) -> Dict:
        """대회 규칙 준수 검증"""
        compliance = {
            "korean_only": True,
            "no_external_api": True,
            "appropriate_content": True,
            "technical_accuracy": True
        }
        
        # 한국어 전용 확인
        import re
        english_chars = len(re.findall(r'[a-zA-Z]', answer))
        total_chars = len(re.sub(r'[^\w가-힣]', '', answer))
        
        if total_chars > 0:
            english_ratio = english_chars / total_chars
            compliance["korean_only"] = english_ratio < 0.1
        
        # 외부 의존성 확인 (URL, 외부 서비스 언급 등)
        external_indicators = ["http", "www", "api", "service", "cloud"]
        compliance["no_external_api"] = not any(indicator in answer.lower() for indicator in external_indicators)
        
        # 도메인 적절성 확인
        if domain in self.domain_keywords:
            domain_keywords = self.domain_keywords[domain]
            found_keywords = sum(1 for keyword in domain_keywords if keyword in answer.lower())
            compliance["appropriate_content"] = found_keywords > 0
        
        return compliance
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()