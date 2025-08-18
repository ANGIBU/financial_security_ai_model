# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 분류
- 전문 용어 처리
- 컨텍스트 정보 제공
- 대회 규칙 준수 검증
- 질문 의도별 지식 제공
- 객관식 패턴 분석 및 컨텍스트 제공
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
from config import JSON_CONFIG_FILES, TEMPLATE_QUALITY_CRITERIA

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # JSON 설정 파일 로드
        self._load_json_configs()
        
        # 템플릿 품질 평가 기준 (config.py에서 로드)
        self.template_quality_criteria = TEMPLATE_QUALITY_CRITERIA
        
        # 객관식 패턴 분석을 위한 추가 데이터
        self._init_enhanced_mc_patterns()
        
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
            "context_usage_stats": {},
            "context_effectiveness": {},
            "mc_pattern_accuracy": {},
            "institution_question_accuracy": {},
            "semantic_pattern_effectiveness": {},
            "choice_categorization_accuracy": {},
            "negative_question_patterns": {},
            "domain_specific_accuracy": {}
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
        
        # 최소한의 기본 설정
        self.korean_subjective_templates = {
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
                ]
            }
        }
        
        self.domain_keywords = {
            "일반": ["법령", "규정", "관리", "조치", "절차"]
        }
        
        self.korean_financial_terms = {}
        self.institution_database = {}
        self.mc_answer_patterns = {}
    
    def _init_enhanced_mc_patterns(self):
        """객관식 패턴 초기화"""
        # 도메인별 세밀한 패턴 분석을 위한 키워드 맵
        self.enhanced_mc_patterns = {
            "금융투자": {
                "업무구분_해당하지않는": {
                    "question_indicators": ["금융투자업", "구분", "해당하지.*않는"],
                    "choice_analysis": {
                        "투자업무": ["투자자문", "투자매매", "투자중개", "집합투자", "신탁"],
                        "비투자업무": ["보험", "소비자금융", "대출", "예금", "카드"],
                        "금융투자법_대상": ["투자자문업", "투자매매업", "투자중개업"],
                        "타법_대상": ["보험중개업", "소비자금융업"]
                    },
                    "context_hint": "금융투자업 카테고리에 속하지 않는 업무를 찾는 문제입니다",
                    "reasoning": "금융투자업 카테고리에 속하지 않는 업무 식별"
                }
            },
            "위험관리": {
                "계획요소_적절하지않은": {
                    "question_indicators": ["위험.*관리", "계획.*수립", "적절하지.*않은"],
                    "choice_analysis": {
                        "계획단계_요소": ["대상", "기간", "범위", "목표", "전략"],
                        "실행단계_요소": ["인력", "자원", "조직", "예산", "담당자"],
                        "대응전략": ["회피", "수용", "전가", "감소"],
                        "관리활동": ["모니터링", "평가", "보고"]
                    },
                    "context_hint": "계획 수립 단계에서 고려하지 않는 실행 리소스를 찾는 문제입니다",
                    "reasoning": "계획 수립 단계에서 고려하지 않는 실행 리소스 식별"
                }
            },
            "개인정보보호": {
                "정책수립_중요요소": {
                    "question_indicators": ["정책.*수립", "가장.*중요한.*요소"],
                    "choice_analysis": {
                        "핵심요소": ["경영진참여", "최고책임자지정", "조직구성"],
                        "절차요소": ["정책제개정", "자원할당", "교육계획"],
                        "운영요소": ["모니터링", "평가", "개선"]
                    },
                    "context_hint": "정책 수립에서 가장 중요한 핵심 요소를 찾는 문제입니다",
                    "reasoning": "정책 수립에서 가장 중요한 핵심 요소 식별"
                }
            },
            "전자금융": {
                "자료제출_요구경우": {
                    "question_indicators": ["한국은행", "자료제출", "요구.*경우"],
                    "choice_analysis": {
                        "법정사유": ["통화신용정책", "지급결제제도", "금융안정"],
                        "비법정사유": ["보안강화", "통계조사", "경영실적", "개인정보"]
                    },
                    "context_hint": "한국은행법에 따른 자료제출 요구 사유를 찾는 문제입니다",
                    "reasoning": "한국은행법에 따른 자료제출 요구 사유 식별"
                }
            },
            "사이버보안": {
                "SBOM_활용이유": {
                    "question_indicators": ["SBOM", "활용.*이유", "적절한"],
                    "choice_analysis": {
                        "주목적": ["소프트웨어공급망보안", "투명성", "취약점관리"],
                        "부차목적": ["접근제어", "개인정보보호", "다양성"]
                    },
                    "context_hint": "SBOM의 주요 활용 목적을 찾는 문제입니다",
                    "reasoning": "SBOM의 주요 활용 목적 식별"
                }
            },
            "정보보안": {
                "재해복구_부적절요소": {
                    "question_indicators": ["재해.*복구", "계획.*수립", "옳지.*않은"],
                    "choice_analysis": {
                        "복구요소": ["복구절차", "비상연락체계", "복구목표시간"],
                        "비복구요소": ["개인정보파기", "일반업무", "성과평가"]
                    },
                    "context_hint": "재해복구와 관련 없는 요소를 찾는 문제입니다",
                    "reasoning": "재해복구와 관련 없는 요소 식별"
                }
            }
        }
    
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
    
    def analyze_question_enhanced(self, question: str) -> Dict:
        """질문 분석"""
        question_lower = question.lower()
        
        # 기본 분석
        basic_analysis = self.analyze_question(question)
        
        # 객관식 패턴 분석
        enhanced_mc_analysis = self._analyze_enhanced_mc_patterns(question, basic_analysis["domain"][0] if basic_analysis["domain"] else "일반")
        
        # 선택지 의미 분석
        choice_semantic_analysis = self._analyze_choice_semantics_advanced(question, basic_analysis["domain"][0] if basic_analysis["domain"] else "일반")
        
        # 부정형 질문 특화 분석
        negative_analysis = self._analyze_negative_question_patterns(question)
        
        # 결과 통합
        enhanced_analysis = {
            **basic_analysis,
            "enhanced_mc_pattern": enhanced_mc_analysis,
            "choice_semantic_analysis": choice_semantic_analysis,
            "negative_analysis": negative_analysis,
            "confidence_score": self._calculate_overall_confidence(basic_analysis, enhanced_mc_analysis, choice_semantic_analysis)
        }
        
        # 이력에 추가
        self._add_to_enhanced_analysis_history(question, enhanced_analysis)
        
        return enhanced_analysis
    
    def _analyze_enhanced_mc_patterns(self, question: str, domain: str) -> Dict:
        """객관식 패턴 분석"""
        pattern_analysis = {
            "matched_pattern": None,
            "pattern_confidence": 0.0,
            "context_hint": "",
            "reasoning": "",
            "choice_analysis": {},
            "domain_specific": False
        }
        
        if domain not in self.enhanced_mc_patterns:
            return pattern_analysis
        
        domain_patterns = self.enhanced_mc_patterns[domain]
        question_lower = question.lower()
        
        for pattern_name, pattern_data in domain_patterns.items():
            # 질문 지표 매칭
            indicator_matches = sum(1 for indicator in pattern_data["question_indicators"] 
                                  if re.search(indicator, question_lower))
            
            if indicator_matches >= 2:  # 최소 2개 지표 매칭
                pattern_analysis["matched_pattern"] = pattern_name
                pattern_analysis["pattern_confidence"] = min(indicator_matches / len(pattern_data["question_indicators"]), 1.0)
                pattern_analysis["context_hint"] = pattern_data["context_hint"]
                pattern_analysis["reasoning"] = pattern_data["reasoning"]
                pattern_analysis["choice_analysis"] = pattern_data["choice_analysis"]
                pattern_analysis["domain_specific"] = True
                break
        
        return pattern_analysis
    
    def _analyze_choice_semantics_advanced(self, question: str, domain: str) -> Dict:
        """고급 선택지 의미 분석"""
        semantic_analysis = {
            "choices_extracted": {},
            "category_mapping": {},
            "outlier_detection": [],
            "semantic_confidence": 0.0,
            "context_suggestion": ""
        }
        
        # 선택지 추출
        choices = self._extract_choices_with_content(question)
        semantic_analysis["choices_extracted"] = choices
        
        if len(choices) < 3:
            return semantic_analysis
        
        # 도메인별 카테고리 매핑
        if domain in self.enhanced_mc_patterns:
            for pattern_name, pattern_data in self.enhanced_mc_patterns[domain].items():
                choice_categories = pattern_data.get("choice_analysis", {})
                
                # 각 선택지를 카테고리에 매핑
                for choice_num, choice_content in choices.items():
                    content_lower = choice_content.lower()
                    best_category = None
                    best_score = 0
                    
                    for category, keywords in choice_categories.items():
                        score = sum(1 for keyword in keywords if keyword in content_lower)
                        if score > best_score:
                            best_score = score
                            best_category = category
                    
                    if best_category:
                        semantic_analysis["category_mapping"][choice_num] = best_category
        
        # 이상치 탐지 (다른 카테고리에 속하는 선택지)
        if semantic_analysis["category_mapping"]:
            category_counts = {}
            for choice_num, category in semantic_analysis["category_mapping"].items():
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # 가장 적은 빈도의 카테고리 찾기
            min_count = min(category_counts.values())
            rare_categories = [cat for cat, count in category_counts.items() if count == min_count]
            
            # 이상치 후보 찾기
            for choice_num, category in semantic_analysis["category_mapping"].items():
                if category in rare_categories:
                    semantic_analysis["outlier_detection"].append(choice_num)
            
            # 컨텍스트 제안
            if semantic_analysis["outlier_detection"] and self._is_negative_question(question):
                semantic_analysis["context_suggestion"] = "부정형 질문에서 다른 카테고리에 속하는 선택지가 탐지되었습니다."
        
        # 신뢰도 계산
        if semantic_analysis["category_mapping"]:
            mapped_count = len(semantic_analysis["category_mapping"])
            total_choices = len(choices)
            semantic_analysis["semantic_confidence"] = mapped_count / total_choices
        
        return semantic_analysis
    
    def _analyze_negative_question_patterns(self, question: str) -> Dict:
        """부정형 질문 패턴 분석"""
        negative_analysis = {
            "is_negative": False,
            "negative_type": None,
            "target_concept": None,
            "exclusion_logic": None,
            "confidence": 0.0,
            "context_guidance": ""
        }
        
        question_lower = question.lower()
        
        # 부정형 패턴 탐지
        negative_patterns = [
            ("해당하지않는", r"해당하지.*않는", "카테고리_예외"),
            ("적절하지않은", r"적절하지.*않은", "부적절_요소"),
            ("옳지않은", r"옳지.*않은", "부정확_내용"),
            ("틀린것", r"틀린.*것", "오류_식별"),
            ("잘못된것", r"잘못된.*것", "잘못된_내용")
        ]
        
        for pattern_name, pattern_regex, exclusion_type in negative_patterns:
            if re.search(pattern_regex, question_lower):
                negative_analysis["is_negative"] = True
                negative_analysis["negative_type"] = pattern_name
                negative_analysis["exclusion_logic"] = exclusion_type
                negative_analysis["confidence"] = 0.8
                negative_analysis["context_guidance"] = f"{exclusion_type}를 찾는 부정형 질문입니다"
                break
        
        # 대상 개념 식별
        if negative_analysis["is_negative"]:
            if "금융투자업" in question_lower and "구분" in question_lower:
                negative_analysis["target_concept"] = "금융투자업_카테고리"
            elif "위험.*관리" in question_lower and "계획" in question_lower:
                negative_analysis["target_concept"] = "위험관리_계획요소"
            elif "재해.*복구" in question_lower and "계획" in question_lower:
                negative_analysis["target_concept"] = "재해복구_계획요소"
            elif "정책.*수립" in question_lower and "요소" in question_lower:
                negative_analysis["target_concept"] = "정책수립_요소"
        
        return negative_analysis
    
    def _calculate_overall_confidence(self, basic_analysis: Dict, enhanced_mc: Dict, semantic: Dict) -> float:
        """전체 신뢰도 계산"""
        confidence_scores = []
        
        # 기본 분석 신뢰도
        if basic_analysis.get("complexity", 0) > 0.5:
            confidence_scores.append(0.7)
        
        # 패턴 신뢰도
        if enhanced_mc.get("pattern_confidence", 0) > 0:
            confidence_scores.append(enhanced_mc["pattern_confidence"])
        
        # 의미 분석 신뢰도
        if semantic.get("semantic_confidence", 0) > 0:
            confidence_scores.append(semantic["semantic_confidence"])
        
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.3
    
    def _extract_choices_with_content(self, question: str) -> Dict[str, str]:
        """선택지 번호와 내용 추출"""
        choices = {}
        lines = question.split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                num = match.group(1)
                content = match.group(2).strip()
                if 1 <= int(num) <= 5 and len(content) > 0:
                    choices[num] = content
        
        return choices
    
    def _is_negative_question(self, question: str) -> bool:
        """부정형 질문 여부 확인"""
        negative_indicators = ["해당하지.*않는", "적절하지.*않은", "옳지.*않은", "틀린", "잘못된"]
        question_lower = question.lower()
        
        return any(re.search(indicator, question_lower) for indicator in negative_indicators)
    
    def _add_to_enhanced_analysis_history(self, question: str, analysis: Dict):
        """분석 이력에 추가"""
        # 기존 이력 추가 로직
        self._add_to_analysis_history(question, analysis)
        
        # 패턴 효과성 기록
        if analysis.get("enhanced_mc_pattern", {}).get("matched_pattern"):
            pattern_name = analysis["enhanced_mc_pattern"]["matched_pattern"]
            if pattern_name not in self.analysis_history["semantic_pattern_effectiveness"]:
                self.analysis_history["semantic_pattern_effectiveness"][pattern_name] = {
                    "usage_count": 0,
                    "high_confidence_count": 0,
                    "avg_confidence": 0.0
                }
            
            effectiveness = self.analysis_history["semantic_pattern_effectiveness"][pattern_name]
            effectiveness["usage_count"] += 1
            
            confidence = analysis["enhanced_mc_pattern"]["pattern_confidence"]
            effectiveness["avg_confidence"] = (
                effectiveness["avg_confidence"] * (effectiveness["usage_count"] - 1) + confidence
            ) / effectiveness["usage_count"]
            
            if confidence > 0.7:
                effectiveness["high_confidence_count"] += 1
        
        # 선택지 카테고리화 정확도 기록
        if analysis.get("choice_semantic_analysis", {}).get("category_mapping"):
            domain = analysis["domain"][0] if analysis["domain"] else "일반"
            if domain not in self.analysis_history["choice_categorization_accuracy"]:
                self.analysis_history["choice_categorization_accuracy"][domain] = {
                    "total_attempts": 0,
                    "successful_categorizations": 0,
                    "outlier_detections": 0
                }
            
            accuracy = self.analysis_history["choice_categorization_accuracy"][domain]
            accuracy["total_attempts"] += 1
            
            if len(analysis["choice_semantic_analysis"]["category_mapping"]) >= 3:
                accuracy["successful_categorizations"] += 1
            
            if analysis["choice_semantic_analysis"]["outlier_detection"]:
                accuracy["outlier_detections"] += 1
        
        # 부정형 질문 패턴 기록
        if analysis.get("negative_analysis", {}).get("is_negative"):
            negative_type = analysis["negative_analysis"]["negative_type"]
            if negative_type not in self.analysis_history["negative_question_patterns"]:
                self.analysis_history["negative_question_patterns"][negative_type] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "target_concepts": {}
                }
            
            pattern_data = self.analysis_history["negative_question_patterns"][negative_type]
            pattern_data["count"] += 1
            
            confidence = analysis["negative_analysis"]["confidence"]
            pattern_data["avg_confidence"] = (
                pattern_data["avg_confidence"] * (pattern_data["count"] - 1) + confidence
            ) / pattern_data["count"]
            
            target_concept = analysis["negative_analysis"]["target_concept"]
            if target_concept:
                pattern_data["target_concepts"][target_concept] = pattern_data["target_concepts"].get(target_concept, 0) + 1
    
    def get_institutional_context(self, institution_type: str) -> str:
        """기관별 컨텍스트 정보 제공 (답변 직접 반환 대신)"""
        
        if institution_type in self.institution_database:
            info = self.institution_database[institution_type]
            
            # 컨텍스트 정보 구성 (직접 답변이 아닌 참고 정보)
            context_parts = []
            
            if "기관명" in info:
                context_parts.append(f"담당 기관: {info['기관명']}")
            
            if "소속" in info:
                context_parts.append(f"소속: {info['소속']}")
            
            if "역할" in info:
                context_parts.append(f"역할: {info['역할']}")
            
            if "근거법" in info:
                context_parts.append(f"근거법: {info['근거법']}")
            
            return " / ".join(context_parts)
        
        # 기본 컨텍스트
        return "해당 분야의 전문 기관에서 업무를 담당합니다"
    
    def get_domain_context(self, domain: str, intent_type: str = "일반") -> str:
        """도메인별 컨텍스트 정보 제공"""
        
        # 컨텍스트 사용 통계 업데이트
        context_key = f"{domain}_{intent_type}"
        if context_key not in self.analysis_history["context_usage_stats"]:
            self.analysis_history["context_usage_stats"][context_key] = 0
        self.analysis_history["context_usage_stats"][context_key] += 1
        
        # 도메인별 참고 정보 구성
        context_parts = []
        
        # 도메인 키워드 정보
        if domain in self.domain_keywords:
            keywords = self.domain_keywords[domain][:5]  # 상위 5개 키워드만
            context_parts.append(f"주요 키워드: {', '.join(keywords)}")
        
        # 한국어 전문 용어 정보
        domain_terms = [term for term in self.korean_financial_terms.keys() 
                       if any(keyword in term for keyword in self.domain_keywords.get(domain, []))][:3]
        if domain_terms:
            context_parts.append(f"전문 용어: {', '.join(domain_terms)}")
        
        # 의도별 지침
        intent_guidance_map = {
            "기관_묻기": "구체적인 기관명과 소속을 포함하여 답변하세요",
            "특징_묻기": "주요 특징과 특성을 체계적으로 설명하세요",
            "지표_묻기": "관찰 가능한 지표와 탐지 방법을 제시하세요",
            "방안_묻기": "실무적이고 실행 가능한 방안을 제시하세요",
            "절차_묻기": "단계별 절차를 순서대로 설명하세요",
            "조치_묻기": "필요한 보안조치와 대응조치를 설명하세요"
        }
        
        if intent_type in intent_guidance_map:
            context_parts.append(intent_guidance_map[intent_type])
        
        return " / ".join(context_parts) if context_parts else "관련 법령과 규정에 따라 답변하세요"
    
    def get_mc_pattern_context(self, question: str) -> str:
        """객관식 패턴 기반 컨텍스트 제공 (답변 직접 반환 대신)"""
        
        question_lower = question.lower()
        context_info = []
        
        # 실제 데이터 패턴 매칭
        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            keyword_matches = sum(1 for keyword in pattern_data["question_keywords"] 
                                if keyword in question_lower)
            
            if keyword_matches >= 2:
                context_info.append(f"문제 유형: {pattern_key}")
                if "explanation" in pattern_data:
                    context_info.append(f"참고: {pattern_data['explanation']}")
                break
        
        return " / ".join(context_info) if context_info else ""
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 (기존 로직 유지)"""
        question_lower = question.lower()
        
        # 도메인 찾기
        detected_domains = []
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드 가중치 적용
                    if keyword in ["트로이", "RAT", "원격제어", "SBOM", "전자금융분쟁조정위원회", 
                                  "개인정보보호위원회", "만 14세", "위험 관리", "금융투자업"]:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # 가장 높은 점수의 도메인 선택
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
        
        # 기관 관련 질문인지 확인
        institution_info = self._check_institution_question(question)
        
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
    
    def _analyze_mc_pattern(self, question: str) -> Dict:
        """객관식 패턴 분석"""
        question_lower = question.lower()
        
        pattern_info = {
            "is_mc_question": False,
            "pattern_type": None,
            "context_hint": None,
            "confidence": 0.0,
            "pattern_key": None
        }
        
        # 실제 데이터 패턴 매칭
        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            keyword_matches = sum(1 for keyword in pattern_data["question_keywords"] 
                                if keyword in question_lower)
            
            if keyword_matches >= 2:
                pattern_info["is_mc_question"] = True
                pattern_info["pattern_type"] = pattern_key
                pattern_info["context_hint"] = pattern_data.get("explanation", "")
                pattern_info["confidence"] = keyword_matches / len(pattern_data["question_keywords"])
                pattern_info["pattern_key"] = pattern_key
                break
        
        return pattern_info
    
    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인"""
        question_lower = question.lower()
        
        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None,
            "context_hint": ""
        }
        
        # 기관 질문 패턴 확인
        institution_patterns = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관",
            "담당.*기관", "관리.*기관", "감독.*기관", "소관.*기관",
            "신고.*기관", "접수.*기관", "상담.*기관", "문의.*기관",
            "위원회.*무엇", "위원회.*어디", "위원회.*설명"
        ]
        
        pattern_matches = 0
        matched_pattern = None
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 1
                matched_pattern = pattern
        
        is_asking_institution = pattern_matches > 0
        
        if is_asking_institution:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 2, 1.0)
            institution_info["question_pattern"] = matched_pattern
            institution_info["context_hint"] = "기관명을 포함한 답변이 필요합니다"
            
            # 분야별 기관 확인
            for institution_key, institution_data in self.institution_database.items():
                if "관련질문패턴" in institution_data:
                    pattern_score = sum(1 for pattern in institution_data["관련질문패턴"] 
                                      if pattern.lower() in question_lower)
                    
                    if pattern_score > 0:
                        institution_info["institution_type"] = institution_key
                        institution_info["relevant_institution"] = institution_data
                        institution_info["confidence"] = min(pattern_score / len(institution_data["관련질문패턴"]), 1.0)
                        break
            
            # 기존 로직으로 폴백
            if not institution_info["institution_type"]:
                if any(word in question_lower for word in ["전자금융", "전자적"]) and "분쟁" in question_lower:
                    institution_info["institution_type"] = "전자금융분쟁조정"
                    institution_info["relevant_institution"] = self.institution_database.get("전자금융분쟁조정", {})
                elif any(word in question_lower for word in ["개인정보", "정보주체"]):
                    institution_info["institution_type"] = "개인정보보호"
                    institution_info["relevant_institution"] = self.institution_database.get("개인정보보호", {})
                elif any(word in question_lower for word in ["금융투자", "투자자문", "자본시장"]) and "분쟁" in question_lower:
                    institution_info["institution_type"] = "금융투자분쟁조정"
                    institution_info["relevant_institution"] = self.institution_database.get("금융투자분쟁조정", {})
                elif any(word in question_lower for word in ["한국은행", "금융통화위원회", "자료제출"]):
                    institution_info["institution_type"] = "한국은행"
                    institution_info["relevant_institution"] = self.institution_database.get("한국은행", {})
        
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
        
        # 객관식 패턴 정확도 추가
        if analysis["mc_pattern_info"]["is_mc_question"]:
            pattern_key = analysis["mc_pattern_info"]["pattern_key"]
            if pattern_key:
                if pattern_key not in self.analysis_history["mc_pattern_accuracy"]:
                    self.analysis_history["mc_pattern_accuracy"][pattern_key] = {
                        "total": 0, "high_confidence": 0
                    }
                
                self.analysis_history["mc_pattern_accuracy"][pattern_key]["total"] += 1
                if analysis["mc_pattern_info"]["confidence"] > 0.7:
                    self.analysis_history["mc_pattern_accuracy"][pattern_key]["high_confidence"] += 1
        
        # 도메인별 정확도 기록
        domain = analysis["domain"][0] if analysis["domain"] else "일반"
        if domain not in self.analysis_history["domain_specific_accuracy"]:
            self.analysis_history["domain_specific_accuracy"][domain] = {
                "total_questions": 0,
                "pattern_matches": 0,
                "high_confidence_answers": 0
            }
        
        domain_accuracy = self.analysis_history["domain_specific_accuracy"][domain]
        domain_accuracy["total_questions"] += 1
        
        if analysis["mc_pattern_info"]["is_mc_question"]:
            domain_accuracy["pattern_matches"] += 1
        
        if analysis["mc_pattern_info"].get("confidence", 0) > 0.7:
            domain_accuracy["high_confidence_answers"] += 1
        
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
    
    def get_domain_specific_guidance(self, domain: str) -> Dict:
        """도메인별 지침 반환"""
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
                "related_institutions": ["전자금융분쟁조정위원회", "금융감독원", "한국은행"],
                "compliance_focus": "한국어 금융 용어 사용",
                "answer_patterns": ["분쟁조정 절차 설명", "기관 역할 명시", "법적 근거 제시"],
                "common_questions": ["분쟁조정 신청 기관", "자료제출 요구 경우"]
            },
            "사이버보안": {
                "key_laws": ["정보통신망법", "개인정보보호법"],
                "key_concepts": ["악성코드", "침입탐지", "보안관제", "사고대응", "트로이", "RAT", "SBOM", "딥페이크"],
                "oversight_body": "과학기술정보통신부, 경찰청",
                "related_institutions": ["한국인터넷진흥원", "사이버보안센터"],
                "compliance_focus": "한국어 보안 용어 사용",
                "answer_patterns": ["탐지 지표 나열", "대응 방안 제시", "특징 상세 설명"],
                "common_questions": ["트로이 목마 특징", "탐지 지표", "SBOM 활용", "딥페이크 대응"]
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
                "related_institutions": ["금융분쟁조정위원회", "금융감독원"],
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
            "context_usage_stats": dict(self.analysis_history["context_usage_stats"]),
            "context_effectiveness": dict(self.analysis_history["context_effectiveness"]),
            "mc_pattern_accuracy": dict(self.analysis_history["mc_pattern_accuracy"]),
            "institution_question_accuracy": dict(self.analysis_history["institution_question_accuracy"]),
            "semantic_pattern_effectiveness": dict(self.analysis_history["semantic_pattern_effectiveness"]),
            "choice_categorization_accuracy": dict(self.analysis_history["choice_categorization_accuracy"]),
            "negative_question_patterns": dict(self.analysis_history["negative_question_patterns"]),
            "domain_specific_accuracy": dict(self.analysis_history["domain_specific_accuracy"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database),
            "context_domains": len(self.korean_subjective_templates),
            "mc_patterns_available": len(self.mc_answer_patterns),
            "enhanced_patterns_available": len(self.enhanced_mc_patterns)
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
            compliance["appropriate_content"] = found_keywords > 0
        
        return compliance
    
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
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()