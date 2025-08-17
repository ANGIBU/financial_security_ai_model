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
    """금융보안 지식베이스 - LLM 힌트 제공 중심"""
    
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
            "hint_usage_stats": {},
            "llm_guidance_provided": 0
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
            print(f"JSON 파일 파싱 오료: {e}")
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
                    "RAT 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하는 원격제어 악성코드입니다."
                ],
                "지표_묻기": [
                    "RAT 악성코드의 주요 탐지 지표로는 비정상적인 네트워크 트래픽, 의심스러운 프로세스 실행 등이 있습니다."
                ],
                "일반": [
                    "사이버보안 위협에 대한 효과적인 대응을 위해 예방, 탐지, 대응, 복구의 단계별 보안 체계를 구축해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "금융감독원 금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다."
                ],
                "일반": [
                    "전자금융거래의 안전성 확보를 위해 관련 법령에 따른 보안 조치를 시행해야 합니다."
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
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 - LLM 힌트 제공용"""
        question_lower = question.lower()
        
        # 도메인 찾기
        detected_domains = []
        domain_scores = {}
        
        # 정확한 도메인 매칭
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
        
        # 복잡도 계산
        complexity = min(len(question) / 200, 1.0)
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 대회 규칙 준수 확인
        compliance_check = self._check_competition_compliance(question)
        
        # 기관 관련 질문인지 확인
        institution_info = self._check_institution_question(question)
        
        # 분석 결과 저장
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info,
            "guidance_provided": True  # LLM 가이드 제공 표시
        }
        
        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def get_institution_hint_for_llm(self, question: str) -> Dict:
        """기관 정보 힌트 반환 (LLM용) - 직접 답변 아님"""
        question_lower = question.lower()
        
        self.analysis_history["hint_usage_stats"]["institution_hints"] = \
            self.analysis_history["hint_usage_stats"].get("institution_hints", 0) + 1
        
        # 기관 유형 감지 및 힌트 제공
        if ("전자금융" in question_lower) and ("분쟁" in question_lower or "조정" in question_lower):
            return {
                "institution_type": "전자금융분쟁조정",
                "institution_name": "금융감독원 금융분쟁조정위원회",
                "parent_organization": "금융감독원",
                "role": "전자금융거래 관련 분쟁조정",
                "legal_basis": "전자금융거래법 제51조",
                "guidance": "금융감독원 산하 기구의 역할과 법적 근거를 포함하여 설명하세요."
            }
        elif ("개인정보" in question_lower) and ("침해" in question_lower or "신고" in question_lower):
            return {
                "institution_type": "개인정보보호",
                "institution_name": "개인정보보호위원회",
                "parent_organization": "국무총리 소속 중앙행정기관",
                "role": "개인정보보호 정책 수립, 감독",
                "legal_basis": "개인정보보호법",
                "guidance": "중앙행정기관의 역할과 개인정보 침해신고 절차를 포함하여 설명하세요."
            }
        elif "한국은행" in question_lower:
            return {
                "institution_type": "한국은행",
                "institution_name": "한국은행",
                "parent_organization": "독립적 중앙은행",
                "role": "통화신용정책 수행, 지급결제제도 운영",
                "legal_basis": "한국은행법",
                "guidance": "중앙은행의 역할과 자료제출 요구권한을 포함하여 설명하세요."
            }
        else:
            return {
                "guidance": "관련 법령에 따른 담당 기관의 역할과 업무를 명확히 설명하세요.",
                "structure_hint": "기관명, 소속, 주요 역할을 포함하여 작성하세요."
            }
    
    def get_content_guidance_for_llm(self, domain: str, intent_type: str = "일반") -> Dict:
        """콘텐츠 가이드라인 제공 (LLM용) - 템플릿 참고용"""
        
        guidance_key = f"{domain}_{intent_type}"
        self.analysis_history["hint_usage_stats"][guidance_key] = \
            self.analysis_history["hint_usage_stats"].get(guidance_key, 0) + 1
        
        self.analysis_history["llm_guidance_provided"] += 1
        
        # 도메인별 콘텐츠 가이드라인
        domain_guidance = {
            "사이버보안": {
                "특징_묻기": {
                    "key_concepts": ["원격제어", "악성코드", "은폐성", "지속성", "시스템 침투"],
                    "structure_guide": "악성코드의 주요 특징과 동작 원리를 체계적으로 설명하세요.",
                    "content_focus": "위장 기법, 원격제어 기능, 지속성 메커니즘을 중심으로 설명하세요.",
                    "template_reference": "정상 프로그램 위장, 시스템 침투, 원격제어 기능 등의 특징 포함"
                },
                "지표_묻기": {
                    "key_concepts": ["네트워크 트래픽", "프로세스 실행", "파일 변조", "레지스트리 변경"],
                    "structure_guide": "탐지 가능한 지표와 징후를 카테고리별로 나열하세요.",
                    "content_focus": "네트워크 활동, 시스템 변경사항, 메모리 패턴을 포함하세요.",
                    "template_reference": "비정상적 네트워크 통신, 의심스러운 프로세스, 시스템 변조"
                },
                "일반": {
                    "key_concepts": ["보안체계", "예방", "탐지", "대응", "복구"],
                    "structure_guide": "사이버보안 대응을 위한 종합적인 체계를 설명하세요.",
                    "content_focus": "단계별 보안 조치와 지속적 모니터링을 포함하세요."
                }
            },
            "전자금융": {
                "기관_묻기": {
                    "key_concepts": ["금융감독원", "분쟁조정위원회", "전자금융거래법"],
                    "structure_guide": "전자금융 분쟁조정 담당 기관과 역할을 명시하세요.",
                    "content_focus": "구체적 기관명, 법적 근거, 주요 업무를 포함하세요.",
                    "template_reference": "금융감독원 금융분쟁조정위원회, 전자금융거래법 제51조"
                },
                "일반": {
                    "key_concepts": ["전자금융거래", "안전성 확보", "이용자 보호", "관리체계"],
                    "structure_guide": "전자금융 안전성을 위한 관리 방안을 설명하세요.",
                    "content_focus": "보안 조치, 이용자 보호, 법령 준수를 포함하세요."
                }
            },
            "개인정보보호": {
                "기관_묻기": {
                    "key_concepts": ["개인정보보호위원회", "개인정보침해신고센터", "개인정보보호법"],
                    "structure_guide": "개인정보 보호 담당 기관과 역할을 명시하세요.",
                    "content_focus": "중앙행정기관 역할, 신고 접수 업무를 포함하세요."
                },
                "일반": {
                    "key_concepts": ["개인정보", "정보주체", "안전성확보조치", "권익보호"],
                    "structure_guide": "개인정보 보호를 위한 관리 방안을 설명하세요.",
                    "content_focus": "법령 준수, 안전성 조치, 정보주체 권리를 포함하세요."
                }
            }
        }
        
        # 해당 도메인과 의도에 맞는 가이드라인 반환
        if domain in domain_guidance:
            domain_guides = domain_guidance[domain]
            if intent_type in domain_guides:
                return domain_guides[intent_type]
            elif "일반" in domain_guides:
                return domain_guides["일반"]
            else:
                return list(domain_guides.values())[0]
        
        # 기본 가이드라인
        return {
            "key_concepts": ["법령", "규정", "관리", "조치", "모니터링"],
            "structure_guide": "관련 법령과 기준에 따른 관리 방안을 설명하세요.",
            "content_focus": "체계적 관리, 지속적 모니터링을 포함하세요.",
            "template_reference": "법령 준수, 관리 체계, 모니터링 수행"
        }
    
    def get_knowledge_context_for_llm(self, question: str, domain: str) -> Dict:
        """LLM을 위한 지식 컨텍스트 제공 - 직접 답변 아님"""
        question_lower = question.lower()
        
        context = {
            "domain": domain,
            "relevant_terms": [],
            "key_patterns": [],
            "structure_hints": [],
            "content_directions": [],
            "quality_guidance": []
        }
        
        # 도메인별 관련 용어 추출
        if domain in self.domain_keywords:
            for keyword in self.domain_keywords[domain]:
                if keyword in question_lower:
                    context["relevant_terms"].append(keyword)
        
        # 특정 패턴별 힌트
        if any(word in question_lower for word in ["rat", "트로이", "원격제어"]):
            if "특징" in question_lower:
                context["key_patterns"].append("RAT_feature_question")
                context["structure_hints"].append("악성코드의 특징을 체계적으로 나열하세요")
                context["content_directions"].append("원격제어 기능과 은폐 기법을 포함하세요")
            elif "지표" in question_lower:
                context["key_patterns"].append("RAT_indicator_question")
                context["structure_hints"].append("탐지 지표를 카테고리별로 설명하세요")
                context["content_directions"].append("네트워크와 시스템 활동 변화를 포함하세요")
        
        if "기관" in question_lower:
            context["key_patterns"].append("institution_question")
            context["structure_hints"].append("구체적인 기관명과 역할을 명시하세요")
            context["content_directions"].append("법적 근거와 주요 업무를 포함하세요")
        
        # 품질 가이드
        context["quality_guidance"] = [
            "한국어 전문 용어를 정확히 사용하세요",
            "법령과 기준을 구체적으로 언급하세요",
            "체계적이고 논리적인 구조로 작성하세요",
            "실무적이고 구체적인 내용을 포함하세요"
        ]
        
        return context
    
    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None
        }
        
        # 기관 질문 패턴
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
                institution_info["confidence"] = 0.9
            elif ("개인정보" in question_lower) and ("침해" in question_lower or "신고" in question_lower):
                institution_info["institution_type"] = "개인정보보호"
                institution_info["confidence"] = 0.8
            elif "한국은행" in question_lower:
                institution_info["institution_type"] = "한국은행"
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
            "hint_usage_stats": dict(self.analysis_history["hint_usage_stats"]),
            "llm_guidance_provided": self.analysis_history["llm_guidance_provided"],
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database),
            "template_domains": len(self.korean_subjective_templates)
        }
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()