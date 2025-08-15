# knowledge_base.py

"""
금융보안 지식베이스
- 한국 금융기관 종합 데이터베이스
- 의도별 특화 템플릿
- 도메인별 답변 시스템
- Self-Consistency 지원
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
        
        # 한국 금융기관 종합 데이터베이스 (강화)
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "전자금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁조정",
                "근거법": "전자금융거래법",
                "신청방법": "금융감독원 홈페이지 또는 방문 신청",
                "관할범위": "전자금융거래 전반",
                "연락처": "금융감독원 내 설치"
            },
            "개인정보보호": {
                "기관명": "개인정보보호위원회",
                "소속": "국무총리 소속",
                "역할": "개인정보보호 정책 수립 및 감시",
                "근거법": "개인정보보호법",
                "신고기관": "개인정보침해신고센터",
                "관할범위": "개인정보 보호 전반",
                "부속기관": "개인정보보호위원회, 개인정보침해신고센터"
            },
            "금융투자분쟁조정": {
                "기관명": "금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "금융투자 관련 분쟁조정",
                "근거법": "자본시장법",
                "관할범위": "금융투자상품 관련 분쟁",
                "신청절차": "금융감독원 통해 신청"
            },
            "금융감독": {
                "기관명": "금융감독원",
                "역할": "금융기관 감독 및 검사",
                "근거법": "금융위원회의 설치 등에 관한 법률",
                "관할범위": "은행, 보험, 증권, 금융투자업 등",
                "주요업무": "금융기관 건전성 감독, 소비자보호"
            },
            "중앙은행": {
                "기관명": "한국은행",
                "역할": "통화정책 수립 및 금융안정",
                "근거법": "한국은행법",
                "관할범위": "통화정책, 금융안정, 지급결제시스템",
                "주요업무": "기준금리 결정, 금융시스템 안정성 관리"
            }
        }
        
        # 시중은행 데이터베이스
        self.commercial_banks = {
            "KB국민은행": {"유형": "시중은행", "설립": "1963년"},
            "신한은행": {"유형": "시중은행", "설립": "1982년"},
            "하나은행": {"유형": "시중은행", "설립": "1971년"},
            "우리은행": {"유형": "시중은행", "설립": "1899년"},
            "농협은행": {"유형": "특수은행", "설립": "1961년"},
            "기업은행": {"유형": "특수은행", "설립": "1961년"}
        }
        
        # 의도별 특화 템플릿 (Self-Consistency 지원)
        self.intent_templates = {
            "기관_요청": {
                "전자금융": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영됩니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                    "전자금융거래법에 따라 금융감독원의 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁의 조정 업무를 담당하고 있습니다."
                ],
                "개인정보보호": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있습니다."
                ],
                "금융투자": [
                    "금융분쟁조정위원회에서 금융투자 관련 분쟁조정 업무를 담당하며, 금융감독원 내에 설치되어 자본시장법에 따라 운영됩니다.",
                    "자본시장법에 따라 금융감독원의 금융분쟁조정위원회에서 금융투자상품 관련 분쟁의 조정 업무를 수행합니다."
                ]
            },
            "특징_분석": {
                "사이버보안": [
                    "트로이 목마 기반 원격접근도구는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "트로이 목마는 유용한 프로그램으로 가장하여 사용자가 직접 설치하도록 유도하고, 설치 후 악의적인 기능을 수행하는 특징을 가집니다. 원격 접근 기능을 통해 시스템을 외부에서 조작할 수 있습니다."
                ]
            },
            "지표_나열": {
                "사이버보안": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다."
                ]
            }
        }
        
        # 도메인별 기본 템플릿
        self.domain_templates = {
            "개인정보보호": [
                "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다.",
                "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다.",
                "전자금융분쟁조정위원회에서 전자금융거래 분쟁조정 업무를 담당하며 이용자는 관련 법령에 따라 분쟁조정을 신청할 수 있습니다."
            ],
            "사이버보안": [
                "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다.",
                "보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다."
            ],
            "정보보안": [
                "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다.",
                "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다."
            ],
            "금융투자": [
                "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다.",
                "금융투자업 영위 시 투자자의 투자성향과 위험도를 평가하고 적합한 상품을 권유하는 적합성 원칙을 준수해야 합니다."
            ]
        }
        
        # 한국어 전문 용어 사전 (확장)
        self.korean_financial_terms = {
            "정보보안관리체계": "조직의 정보자산을 보호하기 위한 종합적인 관리체계",
            "개인정보관리체계": "개인정보의 안전한 처리를 위한 체계적 관리방안",
            "원격접근": "네트워크를 통해 원격지에서 컴퓨터 시스템에 접근하는 방식",
            "지능형지속위협": "특정 목표를 대상으로 장기간에 걸쳐 수행되는 고도화된 사이버공격",
            "전자금융분쟁조정위원회": "전자금융거래 관련 분쟁의 조정을 담당하는 기관",
            "개인정보보호위원회": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관",
            "트로이목마": "정상 프로그램으로 위장하여 악의적 기능을 수행하는 악성코드",
            "원격접근도구": "네트워크를 통해 원격지 시스템을 제어할 수 있는 소프트웨어"
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
            }
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
            
            # 최근 1000개 패턴만 저장
            save_data["question_patterns"] = save_data["question_patterns"][-1000:]
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석"""
        question_lower = question.lower()
        
        # 도메인 찾기
        detected_domains = []
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "개인정보보호법"],
            "전자금융": ["전자금융", "전자금융거래법", "분쟁조정"],
            "사이버보안": ["트로이", "악성코드", "멀웨어", "해킹"],
            "정보보안": ["정보보안", "ISMS", "보안관리"],
            "금융투자": ["금융투자", "자본시장법", "투자자보호"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domains.append(domain)
        
        if not detected_domains:
            detected_domains = ["일반"]
        
        # 복잡도 계산
        complexity = self._calculate_complexity(question)
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 기관 관련 질문인지 확인
        institution_info = self._check_institution_question(question)
        
        # 분석 결과
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "institution_info": institution_info
        }
        
        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None
        }
        
        # 기관을 묻는 질문 패턴
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
    
    def get_korean_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """한국어 주관식 답변 템플릿 반환"""
        
        # 의도별 템플릿 우선 확인
        if intent_type in self.intent_templates:
            intent_templates = self.intent_templates[intent_type]
            if domain in intent_templates:
                return random.choice(intent_templates[domain])
        
        # 도메인별 템플릿 사용
        if domain in self.domain_templates:
            return random.choice(self.domain_templates[domain])
        
        # 기본 템플릿
        default_templates = [
            "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
            "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다.",
            "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다."
        ]
        
        return random.choice(default_templates)
    
    def get_institution_specific_answer(self, institution_type: str) -> str:
        """기관별 구체적 답변 반환"""
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
    
    def get_multiple_templates_for_consistency(self, domain: str, intent_type: str = "일반", count: int = 3) -> List[str]:
        """Self-Consistency를 위한 다중 템플릿 반환"""
        templates = []
        
        # 의도별 템플릿에서 추출
        if intent_type in self.intent_templates and domain in self.intent_templates[intent_type]:
            intent_templates = self.intent_templates[intent_type][domain]
            templates.extend(intent_templates[:count])
        
        # 도메인별 템플릿에서 추가
        if domain in self.domain_templates and len(templates) < count:
            domain_templates = self.domain_templates[domain]
            for template in domain_templates:
                if template not in templates and len(templates) < count:
                    templates.append(template)
        
        # 기본 템플릿으로 채우기
        default_templates = [
            "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
            "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다.",
            "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다."
        ]
        
        for template in default_templates:
            if template not in templates and len(templates) < count:
                templates.append(template)
        
        return templates[:count]
    
    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        # 길이 기반 복잡도
        length_factor = min(len(question) / 200, 1.0)
        
        # 한국어 전문 용어 개수
        korean_term_count = sum(1 for term in self.korean_financial_terms.keys() 
                               if term in question)
        term_factor = min(korean_term_count / 3, 1.0)
        
        # 도메인 개수
        domain_keywords = ["개인정보", "전자금융", "사이버보안", "정보보안", "금융투자"]
        domain_count = sum(1 for keyword in domain_keywords if keyword in question.lower())
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
        if len(analysis["korean_technical_terms"]) > 0:
            self.analysis_history["compliance_check"]["technical_terms"] += 1
        
        # 질문 패턴 추가
        pattern = {
            "question_length": len(question),
            "domain": analysis["domain"][0] if analysis["domain"] else "일반",
            "complexity": analysis["complexity"],
            "korean_terms_count": len(analysis["korean_technical_terms"]),
            "is_institution_question": analysis["institution_info"]["is_institution_question"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history["question_patterns"].append(pattern)
    
    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        return {
            "domain_frequency": dict(self.analysis_history["domain_frequency"]),
            "complexity_distribution": dict(self.analysis_history["complexity_distribution"]),
            "compliance_check": dict(self.analysis_history["compliance_check"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database)
        }
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()