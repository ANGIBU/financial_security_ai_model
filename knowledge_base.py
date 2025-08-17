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
from config import JSON_CONFIG_FILES, TEMPLATE_QUALITY_CRITERIA, TEXT_CLEANUP_CONFIG, KOREAN_TYPO_MAPPING

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스"""
    
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
            "intent_analysis_history": {},
            "template_usage_stats": {},
            "template_effectiveness": {},
            "mc_pattern_accuracy": {},
            "institution_question_accuracy": {},
            "template_quality_improvements": {},
            "korean_text_cleanup_stats": {},
            "typo_correction_stats": {}
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
        
        # 개선된 기본 설정
        self.korean_subjective_templates = {
            "사이버보안": {
                "특징_묻기": [
                    "RAT 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하는 원격제어 악성코드입니다. 주요 특징으로는 은폐성과 지속성을 바탕으로 사용자 모르게 시스템 깊숙이 숨어 장기간 활동하며, 원격제어 기능을 통해 공격자가 외부에서 시스템을 완전히 제어할 수 있습니다. 다양한 악성 기능으로는 키로깅, 화면 캡처, 파일 탈취, 추가 악성코드 다운로드 등이 있으며, 정상 프로그램 위장과 백도어 생성을 통해 탐지를 회피합니다.",
                    "원격접근 트로이목마는 사용자가 알아차리지 못하도록 정상 소프트웨어로 위장하여 배포되며, 설치 후 시스템 권한을 탈취하고 외부 명령제어 서버와 은밀한 통신을 수행합니다. 자동 실행 설정, 프로세스 은닉, 보안 프로그램 무력화 등의 기능을 통해 지속적인 시스템 장악을 유지하며, 모듈식 구조로 필요에 따라 기능을 확장할 수 있습니다."
                ],
                "지표_묻기": [
                    "RAT 악성코드의 주요 탐지 지표로는 네트워크 측면에서 비정상적인 외부 IP로의 주기적 통신, 알려지지 않은 포트 사용, 암호화된 대용량 데이터 전송이 관찰됩니다. 시스템 측면에서는 의심스러운 프로세스 실행, 정상 프로그램명을 사칭한 프로세스, 임시 폴더의 실행 파일 생성이 발견되며, 파일 시스템에서는 레지스트리 자동 실행 항목 추가, 시스템 파일 변조, 숨겨진 파일 생성 등의 흔적이 탐지됩니다.",
                    "네트워크 트래픽 모니터링에서 C2 서버와의 비콘 통신, 비정상적인 DNS 쿼리 패턴, established 상태의 의심스러운 외부 연결이 관찰됩니다. 시스템 성능 저하와 비정상적인 CPU 사용률 증가, 메모리 분석에서 인젝션된 악성 코드와 후킹된 시스템 API가 탐지되며, 이벤트 로그에서 서비스 생성, 작업 스케줄러 등록, PowerShell 실행 흔적이 발견됩니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "금융감독원 금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 전자금융거래법 제51조에 근거하여 설치 운영되며, 전자금융거래와 관련한 분쟁이 발생한 경우 이용자는 금융감독원에 분쟁조정을 신청할 수 있습니다. 조정위원회는 60일 이내에 조정안을 작성하여 당사자에게 제시하며, 조정 전 합의권고 절차를 통해 신속한 분쟁해결을 도모합니다."
                ]
            },
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
                ]
            }
        }
        
        self.domain_keywords = {
            "사이버보안": ["트로이", "RAT", "원격제어", "악성코드", "탐지", "지표"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "전자금융거래법"],
            "일반": ["법령", "규정", "관리", "조치", "절차"]
        }
        
        self.korean_financial_terms = {}
        
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "금융감독원 금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁의 조정",
                "근거법": "전자금융거래법 제51조",
                "관련질문패턴": ["전자금융", "분쟁조정", "신청", "기관"]
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
    
    def clean_template_text_premium(self, text: str) -> str:
        """프리미엄 템플릿 텍스트 정리"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 오류 패턴 감지 및 차단
        error_patterns = [
            r'감추인', r'컨퍼머시', r'피-에', r'백-도어', r'키-로거', r'스크리너',
            r'채팅-클라언트', r'파일-업-', r'[가-힣]-[가-힣]{2,}'
        ]
        
        has_critical_errors = any(re.search(pattern, text) for pattern in error_patterns)
        
        if has_critical_errors:
            # 심각한 오류가 있으면 안전한 기본 템플릿 반환
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
        
        # 안전한 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 - 프리미엄 버전"""
        question_lower = question.lower()
        
        # 도메인 찾기 (정확도 향상)
        detected_domains = []
        domain_scores = {}
        
        # 정확한 도메인 매칭
        domain_patterns = {
            "사이버보안": ["rat", "트로이", "원격제어", "악성코드", "탐지", "지표", "특징", "원격접근"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "전자금융거래법", "분쟁", "조정"],
            "개인정보보호": ["개인정보", "정보주체", "만 14세", "법정대리인", "개인정보보호법"],
            "정보보안": ["정보보안", "isms", "관리체계", "정책 수립", "재해복구"],
            "금융투자": ["금융투자업", "투자자문", "투자매매", "금융투자"],
            "위험관리": ["위험관리", "위험 관리", "재해복구", "위험수용"]
        }
        
        for domain, patterns in domain_patterns.items():
            score = sum(3 if pattern in question_lower else 0 for pattern in patterns)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            detected_domains = [best_domain]
        else:
            detected_domains = ["일반"]
        
        # 복잡도 계산
        complexity = self._calculate_complexity(question)
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 대회 규칙 준수 확인
        compliance_check = self._check_competition_compliance(question)
        
        # 기관 관련 질문인지 확인 (강화)
        institution_info = self._check_institution_question_premium(question)
        
        # 객관식 패턴 매칭
        mc_pattern_info = self._analyze_mc_pattern(question)
        
        # 분석 결과 저장
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info,
            "mc_pattern_info": mc_pattern_info
        }
        
        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def _check_institution_question_premium(self, question: str) -> Dict:
        """기관 관련 질문 확인 - 프리미엄 버전"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None
        }
        
        # 강화된 기관 질문 패턴
        institution_patterns = [
            r"기관.*기술하세요", r"기관.*설명하세요", r"기관.*서술하세요",
            r"어떤.*기관", r"어느.*기관", r"기관.*무엇", r"기관.*어디",
            r"조정.*신청.*기관", r"분쟁.*조정.*기관", r"신청.*수.*있는.*기관",
            r"담당.*기관", r"관리.*기관", r"감독.*기관", r"소관.*기관",
            r"신고.*기관", r"접수.*기관", r"상담.*기관", r"문의.*기관",
            r"위원회.*무엇", r"위원회.*어디", r"위원회.*설명",
            r"분쟁조정.*신청.*가능.*기관", r"침해.*신고.*접수.*기관"
        ]
        
        pattern_matches = 0
        matched_pattern = None
        
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 2
                matched_pattern = pattern
        
        # 추가 기관 키워드 확인
        institution_keywords = [
            "분쟁조정을 신청할 수 있는", "침해신고를 접수하는", "업무를 담당하는",
            "관리하는 기관", "감독하는 기관", "조정 업무", "신고 접수"
        ]
        
        for keyword in institution_keywords:
            if keyword in question_lower:
                pattern_matches += 1
        
        if pattern_matches > 0:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 3, 1.0)
            institution_info["question_pattern"] = matched_pattern
            
            # 정확한 기관 매칭
            if ("전자금융" in question_lower or "전자적" in question_lower) and ("분쟁" in question_lower or "조정" in question_lower):
                institution_info["institution_type"] = "전자금융분쟁조정"
                institution_info["relevant_institution"] = {
                    "기관명": "금융감독원 금융분쟁조정위원회",
                    "소속": "금융감독원",
                    "역할": "전자금융거래 관련 분쟁조정",
                    "근거법": "전자금융거래법 제51조"
                }
                institution_info["confidence"] = 0.95
            elif ("개인정보" in question_lower or "정보주체" in question_lower) and ("침해" in question_lower or "신고" in question_lower):
                institution_info["institution_type"] = "개인정보보호"
                institution_info["relevant_institution"] = {
                    "기관명": "개인정보보호위원회",
                    "소속": "국무총리 소속 중앙행정기관",
                    "역할": "개인정보 보호 업무 총괄",
                    "근거법": "개인정보보호법"
                }
                institution_info["confidence"] = 0.9
            elif "한국은행" in question_lower or "금융통화위원회" in question_lower:
                institution_info["institution_type"] = "한국은행"
                institution_info["relevant_institution"] = {
                    "기관명": "한국은행",
                    "소속": "독립적 중앙은행",
                    "역할": "통화신용정책 수행, 지급결제제도 운영",
                    "근거법": "한국은행법"
                }
                institution_info["confidence"] = 0.9
        
        return institution_info
    
    def _analyze_mc_pattern(self, question: str) -> Dict:
        """객관식 패턴 분석"""
        question_lower = question.lower()
        
        pattern_info = {
            "is_mc_question": False,
            "pattern_type": None,
            "likely_answer": None,
            "confidence": 0.0,
            "pattern_key": None,
            "explanation": ""
        }
        
        # 실제 데이터 패턴 매칭
        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            keyword_matches = sum(1 for keyword in pattern_data["question_keywords"] 
                                if keyword in question_lower)
            
            if keyword_matches >= 2:
                pattern_info["is_mc_question"] = True
                pattern_info["pattern_type"] = pattern_key
                pattern_info["likely_answer"] = pattern_data["correct_answer"]
                pattern_info["confidence"] = keyword_matches / len(pattern_data["question_keywords"])
                pattern_info["pattern_key"] = pattern_key
                pattern_info["explanation"] = pattern_data.get("explanation", "")
                break
        
        return pattern_info
    
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
        
        # 기관 질문 이력 추가
        if analysis["institution_info"]["is_institution_question"]:
            institution_type = analysis["institution_info"]["institution_type"]
            if institution_type:
                if institution_type not in self.analysis_history["institution_question_accuracy"]:
                    self.analysis_history["institution_question_accuracy"][institution_type] = {
                        "total": 0, "high_confidence": 0
                    }
                
                self.analysis_history["institution_question_accuracy"][institution_type]["total"] += 1
                if analysis["institution_info"]["confidence"] > 0.7:
                    self.analysis_history["institution_question_accuracy"][institution_type]["high_confidence"] += 1
        
        # 질문 패턴 추가
        pattern = {
            "question_length": len(question),
            "domain": analysis["domain"][0] if analysis["domain"] else "일반",
            "complexity": analysis["complexity"],
            "korean_terms_count": len(analysis["korean_technical_terms"]),
            "compliance_score": sum(analysis["compliance"].values()) / len(analysis["compliance"]),
            "is_institution_question": analysis["institution_info"]["is_institution_question"],
            "is_mc_pattern": analysis["mc_pattern_info"]["is_mc_question"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history["question_patterns"].append(pattern)
    
    def get_mc_pattern_info(self, question: str) -> Dict:
        """객관식 패턴 정보 반환 (LLM용 힌트)"""
        mc_pattern_info = self._analyze_mc_pattern(question)
        
        if mc_pattern_info["is_mc_question"] and mc_pattern_info["confidence"] > 0.5:
            return {
                "pattern_type": mc_pattern_info["pattern_type"],
                "likely_answer": mc_pattern_info["likely_answer"],
                "confidence": mc_pattern_info["confidence"],
                "explanation": mc_pattern_info.get("explanation", ""),
                "is_reliable": mc_pattern_info["confidence"] > 0.7
            }
        
        return None
    
    def get_institution_hint(self, institution_type: str) -> Dict:
        """기관 정보 힌트 반환 (LLM용) - 프리미엄 버전"""
        
        # 정확한 기관 정보 제공
        institution_data = {
            "전자금융분쟁조정": {
                "institution_name": "금융감독원 금융분쟁조정위원회",
                "parent_organization": "금융감독원",
                "role": "전자금융거래 관련 분쟁조정",
                "legal_basis": "전자금융거래법 제51조",
                "description": "금융감독원 내에 설치된 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 60일 이내에 조정안을 작성하여 당사자에게 제시하며, 조정 전 합의권고 절차를 통해 신속한 분쟁해결을 도모합니다.",
                "contact_info": "전화 1332",
                "specific_duties": ["전자금융거래 분쟁조정", "합의권고", "조정안 작성"]
            },
            "개인정보보호": {
                "institution_name": "개인정보보호위원회",
                "parent_organization": "국무총리 소속 중앙행정기관",
                "role": "개인정보 보호 업무 총괄",
                "legal_basis": "개인정보보호법",
                "description": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관으로 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                "contact_info": "국번없이 118",
                "specific_duties": ["개인정보보호 정책 수립", "침해신고 접수", "실태조사", "개선권고"]
            },
            "한국은행": {
                "institution_name": "한국은행",
                "parent_organization": "독립적 중앙은행",
                "role": "통화신용정책 수행, 지급결제제도 운영",
                "legal_basis": "한국은행법",
                "description": "통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있는 권한을 가집니다.",
                "specific_duties": ["통화신용정책 수행", "지급결제제도 운영", "자료제출 요구"]
            }
        }
        
        if institution_type in institution_data:
            return institution_data[institution_type]
        
        # 기본 힌트
        return {
            "description": "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다.",
            "institution_name": "해당 전문기관",
            "role": "관련 업무 수행"
        }
    
    def get_template_hint(self, domain: str, intent_type: str = "일반") -> str:
        """템플릿 힌트 반환 (LLM용) - 프리미엄 버전"""
        
        # 템플릿 사용 통계 업데이트
        template_key = f"{domain}_{intent_type}"
        if template_key not in self.analysis_history["template_usage_stats"]:
            self.analysis_history["template_usage_stats"][template_key] = 0
        self.analysis_history["template_usage_stats"][template_key] += 1
        
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
                templates = ["관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."]
        
        # 품질 기반 템플릿 선택
        if isinstance(templates, list) and len(templates) > 1:
            # 템플릿 품질 평가 후 선택
            quality_scores = []
            for template in templates:
                # 프리미엄 템플릿 정리
                cleaned_template = self.clean_template_text_premium(template)
                quality = self._evaluate_template_quality_premium(cleaned_template, intent_type)
                quality_scores.append((cleaned_template, quality))
            
            # 상위 품질 템플릿 중에서 선택
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            top_templates = [t for t, q in quality_scores[:3]]
            selected_template = random.choice(top_templates)
        else:
            template = random.choice(templates) if isinstance(templates, list) else templates
            selected_template = self.clean_template_text_premium(template)
        
        # 템플릿 효과성 기록
        if template_key not in self.analysis_history["template_effectiveness"]:
            self.analysis_history["template_effectiveness"][template_key] = {
                "usage_count": 0,
                "avg_length": 0,
                "korean_ratio": 0,
                "quality_score": 0
            }
        
        effectiveness = self.analysis_history["template_effectiveness"][template_key]
        effectiveness["usage_count"] += 1
        effectiveness["avg_length"] = (effectiveness["avg_length"] * (effectiveness["usage_count"] - 1) + len(selected_template)) / effectiveness["usage_count"]
        
        korean_chars = len(re.findall(r'[가-힣]', selected_template))
        total_chars = len(re.sub(r'[^\w가-힣]', '', selected_template))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        effectiveness["korean_ratio"] = (effectiveness["korean_ratio"] * (effectiveness["usage_count"] - 1) + korean_ratio) / effectiveness["usage_count"]
        
        # 품질 점수 기록
        quality_score = self._evaluate_template_quality_premium(selected_template, intent_type)
        effectiveness["quality_score"] = (effectiveness["quality_score"] * (effectiveness["usage_count"] - 1) + quality_score) / effectiveness["usage_count"]
        
        return selected_template
    
    def _evaluate_template_quality_premium(self, template: str, intent_type: str) -> float:
        """템플릿 품질 평가 - 프리미엄 버전"""
        score = 0.0
        
        # 오류 패턴 검증 (-0.5점)
        error_patterns = [
            r'감추인', r'컨퍼머시', r'피-에', r'백-도어', r'키-로거', r'스크리너'
        ]
        
        has_errors = any(re.search(pattern, template) for pattern in error_patterns)
        if has_errors:
            return 0.0  # 오류가 있으면 품질 0점
        
        # 길이 적절성 (25%)
        length = len(template)
        if 50 <= length <= 400:
            score += 0.25
        elif 30 <= length < 50:
            score += 0.20
        elif length > 400:
            score += 0.15
        
        # 한국어 비율 (30%)
        korean_chars = len(re.findall(r'[가-힣]', template))
        total_chars = len(re.sub(r'[^\w가-힣]', '', template))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        if korean_ratio >= 0.9:
            score += 0.30
        elif korean_ratio >= 0.8:
            score += 0.25
        else:
            score += korean_ratio * 0.30
        
        # 구조적 키워드 포함 (25%)
        structure_keywords = ["법령", "규정", "조치", "관리", "절차", "기준", "정책", "체계"]
        found_structure = sum(1 for keyword in structure_keywords if keyword in template)
        score += min(found_structure / len(structure_keywords), 1.0) * 0.25
        
        # 의도별 키워드 포함 (20%)
        intent_keywords = {
            "기관_묻기": ["위원회", "기관", "담당", "업무"],
            "특징_묻기": ["특징", "특성", "성질", "기능"],
            "지표_묻기": ["지표", "징후", "패턴", "탐지"],
            "방안_묻기": ["방안", "대책", "조치", "관리"],
            "절차_묻기": ["절차", "과정", "단계", "순서"],
            "조치_묻기": ["조치", "대응", "보안", "예방"]
        }
        
        if intent_type in intent_keywords:
            intent_words = intent_keywords[intent_type]
            found_intent = sum(1 for keyword in intent_words if keyword in template)
            score += min(found_intent / len(intent_words), 1.0) * 0.20
        else:
            score += 0.15
        
        return min(score, 1.0)
    
    def get_korean_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """한국어 주관식 답변 템플릿 반환 (기존 호환성 유지, LLM 힌트용)"""
        return self.get_template_hint(domain, intent_type)
    
    def get_institution_specific_answer(self, institution_type: str) -> str:
        """기관별 구체적 답변 반환 (기존 호환성 유지, LLM 힌트용)"""
        hint = self.get_institution_hint(institution_type)
        return hint.get("description", "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다.")
    
    def get_mc_pattern_answer(self, question: str) -> str:
        """객관식 패턴 기반 답변 반환 (기존 호환성 유지, LLM 힌트용)"""
        pattern_info = self.get_mc_pattern_info(question)
        
        if pattern_info and pattern_info.get("is_reliable", False):
            return pattern_info.get("likely_answer")
        
        return None
    
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
        """도메인별 지침 반환 - 프리미엄 버전"""
        guidance = {
            "개인정보보호": {
                "key_laws": ["개인정보보호법", "정보통신망법"],
                "key_concepts": ["정보주체", "개인정보처리자", "동의", "목적외이용금지", "만 14세 미만", "법정대리인"],
                "oversight_body": "개인정보보호위원회",
                "related_institutions": ["개인정보보호위원회", "개인정보침해신고센터"],
                "compliance_focus": "한국어 법령 용어 사용",
                "answer_patterns": ["법적 근거 제시", "기관명 정확 명시", "절차 단계별 설명"],
                "common_questions": ["만 14세 미만 아동 동의", "정책 수립 중요 요소", "개인정보 관리체계"]
            },
            "전자금융": {
                "key_laws": ["전자금융거래법", "전자서명법"],
                "key_concepts": ["접근매체", "전자서명", "인증", "분쟁조정", "이용자", "자료제출"],
                "oversight_body": "금융감독원, 한국은행",
                "related_institutions": ["금융감독원 금융분쟁조정위원회", "금융감독원", "한국은행"],
                "compliance_focus": "한국어 금융 용어 사용",
                "answer_patterns": ["분쟁조정 절차 설명", "기관 역할 명시", "법적 근거 제시"],
                "common_questions": ["분쟁조정 신청 기관", "자료제출 요구 경우"]
            },
            "사이버보안": {
                "key_laws": ["정보통신망법", "개인정보보호법"],
                "key_concepts": ["악성코드", "침입탐지", "보안관제", "사고대응", "RAT", "원격제어", "딥페이크"],
                "oversight_body": "과학기술정보통신부, 경찰청",
                "related_institutions": ["한국인터넷진흥원", "사이버보안센터"],
                "compliance_focus": "한국어 보안 용어 사용",
                "answer_patterns": ["탐지 지표 나열", "대응 방안 제시", "특징 상세 설명"],
                "common_questions": ["RAT 특징", "탐지 지표", "딥페이크 대응"]
            },
            "정보보안": {
                "key_laws": ["정보통신망법", "전자정부법"],
                "key_concepts": ["정보보안관리체계", "접근통제", "암호화", "백업", "재해복구"],
                "oversight_body": "과학기술정보통신부",
                "related_institutions": ["한국인터넷진흥원"],
                "compliance_focus": "한국어 기술 용어 사용",
                "answer_patterns": ["관리체계 설명", "보안조치 나열", "절차 단계 제시"],
                "common_questions": ["재해복구 계획", "관리체계 수립"]
            },
            "금융투자": {
                "key_laws": ["자본시장법", "금융투자업규정"],
                "key_concepts": ["투자자보호", "적합성원칙", "설명의무", "내부통제", "금융투자업 구분"],
                "oversight_body": "금융감독원, 금융위원회",
                "related_institutions": ["금융감독원", "금융위원회"],
                "compliance_focus": "한국어 투자 용어 사용",
                "answer_patterns": ["법령 근거 제시", "원칙 설명", "보호 방안 나열"],
                "common_questions": ["금융투자업 구분", "해당하지 않는 업무"]
            },
            "위험관리": {
                "key_laws": ["은행법", "보험업법", "자본시장법"],
                "key_concepts": ["위험평가", "내부통제", "컴플라이언스", "감사", "위험 관리 계획", "재해 복구"],
                "oversight_body": "금융감독원",
                "related_institutions": ["금융감독원"],
                "compliance_focus": "한국어 관리 용어 사용",
                "answer_patterns": ["위험관리 절차", "평가 방법", "대응 체계"],
                "common_questions": ["위험관리 요소", "재해복구 계획", "적절하지 않은 요소"]
            }
        }
        
        return guidance.get(domain, {
            "key_laws": ["관련 법령"],
            "key_concepts": ["체계적 관리", "지속적 개선"],
            "oversight_body": "관계기관",
            "related_institutions": ["해당 전문기관"],
            "compliance_focus": "한국어 전용 답변",
            "answer_patterns": ["법령 근거", "관리 방안", "절차 설명"],
            "common_questions": []
        })
    
    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        return {
            "domain_frequency": dict(self.analysis_history["domain_frequency"]),
            "complexity_distribution": dict(self.analysis_history["complexity_distribution"]),
            "compliance_check": dict(self.analysis_history["compliance_check"]),
            "intent_analysis_history": dict(self.analysis_history["intent_analysis_history"]),
            "template_usage_stats": dict(self.analysis_history["template_usage_stats"]),
            "template_effectiveness": dict(self.analysis_history["template_effectiveness"]),
            "mc_pattern_accuracy": dict(self.analysis_history["mc_pattern_accuracy"]),
            "institution_question_accuracy": dict(self.analysis_history["institution_question_accuracy"]),
            "template_quality_improvements": dict(self.analysis_history["template_quality_improvements"]),
            "korean_text_cleanup_stats": dict(self.analysis_history["korean_text_cleanup_stats"]),
            "typo_correction_stats": dict(self.analysis_history["typo_correction_stats"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database),
            "template_domains": len(self.korean_subjective_templates),
            "mc_patterns_available": len(self.mc_answer_patterns)
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
        
        # 외부 의존성 확인
        external_indicators = ["http", "www", "api", "service", "cloud"]
        compliance["no_external_api"] = not any(indicator in answer.lower() for indicator in external_indicators)
        
        # 도메인 적절성 확인
        if domain in self.domain_keywords:
            domain_keywords = self.domain_keywords[domain]
            found_keywords = sum(1 for keyword in domain_keywords if keyword in answer.lower())
            compliance["appropriate_content"] = found_keywords >= 0
        
        return compliance
    
    def get_high_quality_template(self, domain: str, intent_type: str, min_quality: float = 0.8) -> str:
        """고품질 템플릿 반환 (LLM 힌트용) - 프리미엄 버전"""
        template_key = f"{domain}_{intent_type}"
        
        # 효과성이 검증된 템플릿 우선 사용
        if template_key in self.analysis_history["template_effectiveness"]:
            effectiveness = self.analysis_history["template_effectiveness"][template_key]
            if (effectiveness["korean_ratio"] >= min_quality and 
                effectiveness["usage_count"] >= 2 and
                effectiveness.get("quality_score", 0) >= min_quality):
                # 검증된 고품질 템플릿 사용
                return self.get_template_hint(domain, intent_type)
        
        # 기본 템플릿 반환
        return self.get_template_hint(domain, intent_type)
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()