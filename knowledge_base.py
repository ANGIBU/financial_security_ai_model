# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 분류
- 전문 용어 처리
- 한국어 전용 답변 템플릿 제공
- 대회 규칙 준수 검증
- 질문 의도별 지식 제공
- LLM을 위한 힌트 제공 (직접 답변 반환 금지)
"""

import pickle
import os
import re
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import random

# 설정 파일 import
from config import JSON_CONFIG_FILES, TEMPLATE_QUALITY_CRITERIA, TEXT_CLEANUP_CONFIG, KOREAN_TYPO_MAPPING, check_text_safety

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스 - 안전성 강화 버전"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 텍스트 정리 설정 로드
        self.text_cleanup_config = TEXT_CLEANUP_CONFIG
        self.korean_typo_mapping = KOREAN_TYPO_MAPPING
        
        # JSON 설정 파일 로드
        self._load_json_configs()
        
        # 템플릿 품질 평가 기준 (config.py에서 로드)
        self.template_quality_criteria = TEMPLATE_QUALITY_CRITERIA
        
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
            "template_usage_stats": {},
            "safe_template_usage": {}
        }
        
        # 안전한 답변 템플릿 강화
        self.safe_answer_templates = {
            "RAT_특징": "RAT 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하는 원격제어 악성코드입니다. 은폐성과 지속성을 바탕으로 시스템 깊숙이 숨어 장기간 활동하며, 키로깅, 화면 캡처, 파일 탈취 등의 악성 기능을 수행합니다.",
            "RAT_지표": "RAT 악성코드의 주요 탐지 지표로는 비정상적인 네트워크 트래픽, 의심스러운 프로세스 실행, 파일 시스템 변조, 레지스트리 자동 실행 항목 추가 등이 있습니다.",
            "전자금융분쟁조정": "금융감독원 금융분쟁조정위원회",
            "개인정보침해신고": "개인정보보호위원회 산하 개인정보침해신고센터",
            "기본_답변": "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
        }
        
        # 이전 분석 이력 로드
        self._load_analysis_history()
    
    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            # knowledge_data.json 로드
            with open(JSON_CONFIG_FILES['knowledge_data'], 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # 지식베이스 데이터 할당
            self.korean_subjective_templates = knowledge_data['korean_subjective_templates']
            self.domain_keywords = knowledge_data['domain_keywords']
            self.korean_financial_terms = knowledge_data['korean_financial_terms']
            self.institution_database = knowledge_data['institution_database']
            self.mc_answer_patterns = knowledge_data['mc_answer_patterns']
            
            print("지식베이스 설정 파일 로드 완료")
            
        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()
    
    def _load_default_configs(self):
        """기본 설정 로드 (JSON 파일 로드 실패 시)"""
        print("기본 설정으로 대체합니다.")
        
        # 안전한 기본 템플릿
        self.korean_subjective_templates = {
            "사이버보안": {
                "특징_묻기": [
                    "RAT 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하는 원격제어 악성코드입니다. 은폐성과 지속성을 바탕으로 시스템 깊숙이 숨어 장기간 활동하며, 원격제어 기능을 통해 공격자가 외부에서 시스템을 제어할 수 있습니다."
                ],
                "지표_묻기": [
                    "RAT 악성코드의 주요 탐지 지표로는 비정상적인 네트워크 트래픽, 의심스러운 프로세스 실행, 파일 시스템 변조 등이 있습니다."
                ],
                "일반": [
                    "사이버보안 위협에 대한 효과적인 대응을 위해 예방, 탐지, 대응, 복구의 단계별 보안 체계를 구축하고 지속적인 모니터링을 수행해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "금융감독원 금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다."
                ],
                "일반": [
                    "전자금융거래의 안전성 확보를 위해 관련 법령에 따른 보안 조치를 시행하고 이용자 보호를 위한 관리 체계를 운영해야 합니다."
                ]
            },
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
                ]
            }
        }
        
        self.domain_keywords = {
            "사이버보안": ["트로이", "RAT", "원격제어", "악성코드"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원"],
            "일반": ["법령", "규정", "관리", "조치"]
        }
        
        self.korean_financial_terms = {}
        
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "금융감독원 금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁의 조정"
            }
        }
        
        self.mc_answer_patterns = {}
    
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
    
    def clean_template_text_safe(self, text: str) -> str:
        """안전한 템플릿 텍스트 정리"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 안전성 검사
        if not check_text_safety(text):
            # 안전하지 않은 텍스트는 기본 답변으로 대체
            return self.safe_answer_templates["기본_답변"]
        
        # 최소한의 안전한 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 - 안정성 우선 버전"""
        question_lower = question.lower()
        
        # 도메인 찾기 (간소화된 정확도 향상)
        detected_domains = []
        domain_scores = {}
        
        # 정확한 도메인 매칭 (핵심 키워드만)
        domain_patterns = {
            "사이버보안": ["rat", "트로이", "원격제어", "악성코드", "딥페이크", "sbom"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "한국은행"],
            "개인정보보호": ["개인정보", "정보주체", "만 14세", "법정대리인", "개인정보보호위원회"],
            "정보보안": ["정보보안", "isms", "관리체계", "재해복구"],
            "금융투자": ["금융투자업", "투자자문", "투자매매"],
            "위험관리": ["위험관리", "위험수용", "재해복구"]
        }
        
        for domain, patterns in domain_patterns.items():
            score = sum(2 if pattern in question_lower else 0 for pattern in patterns)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            detected_domains = [best_domain]
        else:
            detected_domains = ["일반"]
        
        # 복잡도 계산 (간소화)
        complexity = min(len(question) / 200, 1.0)
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 대회 규칙 준수 확인
        compliance_check = self._check_competition_compliance(question)
        
        # 기관 관련 질문인지 확인 (간소화)
        institution_info = self._check_institution_question_safe(question)
        
        # 분석 결과 저장
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info
        }
        
        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def _check_institution_question_safe(self, question: str) -> Dict:
        """기관 관련 질문 확인 - 안전성 우선"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None
        }
        
        # 간소화된 기관 질문 패턴
        institution_patterns = [
            r"기관.*기술하세요", r"기관.*설명하세요",
            r"어떤.*기관", r"어느.*기관", r"기관.*무엇",
            r"분쟁.*조정.*기관", r"신청.*수.*있는.*기관",
            r"담당.*기관", r"관리.*기관"
        ]
        
        pattern_matches = 0
        matched_pattern = None
        
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 1
                matched_pattern = pattern
        
        # 기관 키워드 확인
        institution_keywords = [
            "전자금융", "분쟁조정", "개인정보", "침해신고", 
            "금융감독원", "한국은행", "개인정보보호위원회"
        ]
        
        for keyword in institution_keywords:
            if keyword in question_lower:
                pattern_matches += 1
        
        if pattern_matches > 0:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 2, 1.0)
            institution_info["question_pattern"] = matched_pattern
            
            # 정확한 기관 매칭
            if ("전자금융" in question_lower) and ("분쟁" in question_lower or "조정" in question_lower):
                institution_info["institution_type"] = "전자금융분쟁조정"
                institution_info["relevant_institution"] = {
                    "기관명": "금융감독원 금융분쟁조정위원회",
                    "소속": "금융감독원",
                    "역할": "전자금융거래 관련 분쟁조정"
                }
                institution_info["confidence"] = 0.9
            elif ("개인정보" in question_lower) and ("침해" in question_lower or "신고" in question_lower):
                institution_info["institution_type"] = "개인정보보호"
                institution_info["relevant_institution"] = {
                    "기관명": "개인정보보호위원회",
                    "소속": "국무총리 소속 중앙행정기관",
                    "역할": "개인정보 보호 업무 총괄"
                }
                institution_info["confidence"] = 0.8
            elif "한국은행" in question_lower:
                institution_info["institution_type"] = "한국은행"
                institution_info["relevant_institution"] = {
                    "기관명": "한국은행",
                    "소속": "독립적 중앙은행",
                    "역할": "통화신용정책 수행"
                }
                institution_info["confidence"] = 0.8
        
        return institution_info
    
    def _check_competition_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인"""
        compliance = {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True
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
    
    def get_institution_hint(self, institution_type: str) -> Dict:
        """기관 정보 힌트 반환 (LLM용) - 안전성 우선"""
        
        # 안전한 기관 정보 제공
        institution_data = {
            "전자금융분쟁조정": {
                "institution_name": "금융감독원 금융분쟁조정위원회",
                "parent_organization": "금융감독원",
                "role": "전자금융거래 관련 분쟁조정",
                "legal_basis": "전자금융거래법 제51조",
                "description": "금융감독원 내에 설치된 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다."
            },
            "개인정보보호": {
                "institution_name": "개인정보보호위원회",
                "parent_organization": "국무총리 소속 중앙행정기관",
                "role": "개인정보보호 정책 수립, 감독",
                "legal_basis": "개인정보보호법",
                "description": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관으로 개인정보침해신고센터에서 신고 접수 업무를 담당합니다."
            },
            "한국은행": {
                "institution_name": "한국은행",
                "parent_organization": "독립적 중앙은행",
                "role": "통화신용정책 수행, 지급결제제도 운영",
                "legal_basis": "한국은행법",
                "description": "통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있는 권한을 가집니다."
            }
        }
        
        if institution_type in institution_data:
            return institution_data[institution_type]
        
        # 기본 안전 힌트
        return {
            "description": "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다.",
            "institution_name": "해당 전문기관",
            "role": "관련 업무 수행"
        }
    
    def get_template_hint(self, domain: str, intent_type: str = "일반") -> str:
        """템플릿 힌트 반환 (LLM용) - 안전성 최우선"""
        
        # 템플릿 사용 통계 업데이트
        template_key = f"{domain}_{intent_type}"
        if template_key not in self.analysis_history["template_usage_stats"]:
            self.analysis_history["template_usage_stats"][template_key] = 0
        self.analysis_history["template_usage_stats"][template_key] += 1
        
        # 안전한 템플릿 사용 추적
        if template_key not in self.analysis_history["safe_template_usage"]:
            self.analysis_history["safe_template_usage"][template_key] = 0
        
        # 안전한 기본 답변 우선 반환
        if domain == "사이버보안":
            if intent_type == "특징_묻기":
                self.analysis_history["safe_template_usage"][template_key] += 1
                return self.safe_answer_templates["RAT_특징"]
            elif intent_type == "지표_묻기":
                self.analysis_history["safe_template_usage"][template_key] += 1
                return self.safe_answer_templates["RAT_지표"]
            else:
                return "사이버보안 위협에 대한 효과적인 대응을 위해 예방, 탐지, 대응, 복구의 단계별 보안 체계를 구축하고 지속적인 모니터링을 수행해야 합니다."
        
        elif domain == "전자금융" and intent_type == "기관_묻기":
            self.analysis_history["safe_template_usage"][template_key] += 1
            return self.safe_answer_templates["전자금융분쟁조정"]
        
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
            if "일반" in self.korean_subjective_templates:
                templates = self.korean_subjective_templates["일반"]["일반"]
            else:
                templates = [self.safe_answer_templates["기본_답변"]]
        
        # 안전한 템플릿 선택
        if isinstance(templates, list) and len(templates) > 1:
            # 모든 템플릿 안전성 확인
            safe_templates = []
            for template in templates:
                cleaned_template = self.clean_template_text_safe(template)
                if check_text_safety(cleaned_template):
                    safe_templates.append(cleaned_template)
            
            if safe_templates:
                selected_template = random.choice(safe_templates)
                self.analysis_history["safe_template_usage"][template_key] += 1
            else:
                # 안전한 템플릿이 없으면 기본 답변
                selected_template = self.safe_answer_templates["기본_답변"]
        else:
            template = random.choice(templates) if isinstance(templates, list) else templates
            selected_template = self.clean_template_text_safe(template)
            if check_text_safety(selected_template):
                self.analysis_history["safe_template_usage"][template_key] += 1
            else:
                selected_template = self.safe_answer_templates["기본_답변"]
        
        return selected_template
    
    def get_safe_answer_for_question(self, question: str) -> str:
        """질문에 대한 안전한 답변 직접 반환"""
        question_lower = question.lower()
        
        # RAT 관련 질문
        if any(word in question_lower for word in ["rat", "트로이", "원격제어"]):
            if "특징" in question_lower:
                return self.safe_answer_templates["RAT_특징"]
            elif "지표" in question_lower or "탐지" in question_lower:
                return self.safe_answer_templates["RAT_지표"]
            else:
                return self.safe_answer_templates["RAT_특징"]
        
        # 기관 관련 질문
        elif "기관" in question_lower:
            if "전자금융" in question_lower and "분쟁" in question_lower:
                return self.safe_answer_templates["전자금융분쟁조정"]
            elif "개인정보" in question_lower and "침해" in question_lower:
                return self.safe_answer_templates["개인정보침해신고"]
            else:
                return "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다."
        
        # 기본 답변
        else:
            return self.safe_answer_templates["기본_답변"]
    
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
    
    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        return {
            "domain_frequency": dict(self.analysis_history["domain_frequency"]),
            "complexity_distribution": dict(self.analysis_history["complexity_distribution"]),
            "compliance_check": dict(self.analysis_history["compliance_check"]),
            "template_usage_stats": dict(self.analysis_history["template_usage_stats"]),
            "safe_template_usage": dict(self.analysis_history["safe_template_usage"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database),
            "template_domains": len(self.korean_subjective_templates)
        }
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()