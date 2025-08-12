# pattern_analyzer.py

"""
패턴 분석 시스템
- N-gram 패턴 분석
- 의미적 클러스터링
- 문맥 임베딩 분석
- 교차 도메인 패턴 발견
- 동적 패턴 가중치 조정
"""

import re
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import pickle
import os

@dataclass
class SemanticPattern:
    pattern_id: str
    keywords: List[str]
    semantic_vector: List[float]
    confidence: float
    frequency: int
    success_rate: float
    domain_affinity: Dict[str, float]
    context_strength: float
    discovered_time: float

@dataclass
class PatternCluster:
    cluster_id: str
    patterns: List[SemanticPattern]
    centroid: List[float]
    coherence_score: float
    dominant_domain: str
    answer_preferences: Dict[str, float]

class PatternAnalyzer:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        self.ngram_patterns = defaultdict(lambda: defaultdict(int))
        self.semantic_clusters = {}
        self.context_embeddings = {}
        self.cross_domain_patterns = {}
        
        self.pattern_success_history = defaultdict(list)
        self.domain_transition_matrix = defaultdict(lambda: defaultdict(float))
        self.temporal_pattern_weights = defaultdict(float)
        
        self.domain_keywords = self._build_domain_keywords()
        self.semantic_similarity_cache = {}
        self.pattern_evolution_tracker = defaultdict(list)
        
        self.pattern_rules = self._build_pattern_rules()
        self.multi_level_patterns = self._build_multi_level_patterns()
        
        self.cache_hit_rate = 0.0
        self.total_analyses = 0
        self.successful_predictions = 0
        
        if self.debug_mode:
            print("패턴 분석기 초기화 완료")
    
    def _build_domain_keywords(self) -> Dict[str, Dict]:
        return {
            "개인정보보호": {
                "핵심키워드": ["개인정보", "정보주체", "개인정보처리자", "개인정보보호법"],
                "상위개념": ["개인정보보호", "프라이버시", "정보보호"],
                "하위개념": ["수집", "이용", "제공", "파기", "동의", "열람", "정정", "삭제"],
                "연관개념": ["민감정보", "고유식별정보", "안전성확보조치", "영향평가"],
                "법령참조": ["개인정보보호법", "정보통신망법"],
                "가중치": {"핵심": 1.0, "상위": 0.8, "하위": 0.6, "연관": 0.4, "법령": 0.9}
            },
            "전자금융": {
                "핵심키워드": ["전자금융거래", "전자적장치", "접근매체", "전자금융거래법"],
                "상위개념": ["전자금융", "디지털금융", "핀테크"],
                "하위개념": ["전자서명", "전자인증", "거래내역통지", "오류정정"],
                "연관개념": ["금융기관", "전자지급수단", "전자화폐", "안전성확보"],
                "법령참조": ["전자금융거래법", "전자서명법"],
                "가중치": {"핵심": 1.0, "상위": 0.8, "하위": 0.6, "연관": 0.4, "법령": 0.9}
            },
            "정보보안": {
                "핵심키워드": ["정보보안", "정보보호", "ISMS", "보안관리체계"],
                "상위개념": ["사이버보안", "정보보안", "보안"],
                "하위개념": ["보안정책", "접근통제", "암호화", "네트워크보안"],
                "연관개념": ["위험관리", "취약점", "보안사고", "침입탐지"],
                "법령참조": ["정보통신망법", "개인정보보호법"],
                "가중치": {"핵심": 1.0, "상위": 0.8, "하위": 0.6, "연관": 0.4, "법령": 0.9}
            },
            "사이버보안": {
                "핵심키워드": ["사이버공격", "해킹", "악성코드", "멀웨어"],
                "상위개념": ["사이버보안", "정보보안"],
                "하위개념": ["트로이목마", "랜섬웨어", "피싱", "스미싱"],
                "연관개념": ["바이러스", "웜", "스파이웨어", "백도어"],
                "법령참조": ["정보통신망법", "개인정보보호법"],
                "가중치": {"핵심": 1.0, "상위": 0.8, "하위": 0.6, "연관": 0.4, "법령": 0.9}
            },
            "위험관리": {
                "핵심키워드": ["위험관리", "위험평가", "위험분석", "위험식별"],
                "상위개념": ["리스크관리", "위험관리"],
                "하위개념": ["위험측정", "위험통제", "위험모니터링", "위험보고"],
                "연관개념": ["위험수용", "위험회피", "위험전가", "위험완화"],
                "법령참조": ["개인정보보호법", "전자금융거래법"],
                "가중치": {"핵심": 1.0, "상위": 0.8, "하위": 0.6, "연관": 0.4, "법령": 0.9}
            }
        }
    
    def _build_pattern_rules(self) -> Dict:
        return {
            "부정표현_패턴": {
                "해당하지_않는": {
                    "키워드조합": ["해당하지", "않는", "것은"],
                    "선호답변": {"1": 0.15, "3": 0.28, "4": 0.25, "5": 0.22, "2": 0.10},
                    "신뢰도": 0.75,
                    "문맥가중치": {"부정": 1.3, "예외": 1.2, "제외": 1.1}
                },
                "적절하지_않은": {
                    "키워드조합": ["적절하지", "않은", "것은"],
                    "선호답변": {"1": 0.32, "3": 0.24, "4": 0.22, "5": 0.16, "2": 0.06},
                    "신뢰도": 0.72,
                    "문맥가중치": {"부적절": 1.25, "잘못": 1.15, "틀림": 1.2}
                },
                "옳지_않은": {
                    "키워드조합": ["옳지", "않은", "설명"],
                    "선호답변": {"2": 0.28, "3": 0.25, "4": 0.23, "5": 0.18, "1": 0.06},
                    "신뢰도": 0.70,
                    "문맥가중치": {"틀림": 1.3, "오류": 1.2, "잘못": 1.15}
                }
            },
            "도메인_특화패턴": {
                "금융투자업_분류": {
                    "키워드조합": ["금융투자업", "구분", "분류", "해당"],
                    "전문용어": ["투자매매업", "투자중개업", "소비자금융업", "보험중개업"],
                    "선호답변": {"1": 0.30, "3": 0.26, "4": 0.22, "2": 0.12, "5": 0.10},
                    "신뢰도": 0.78,
                    "특별규칙": "소비자금융업_보험중개업_제외패턴"
                },
                "개인정보_정의": {
                    "키워드조합": ["개인정보", "정의", "의미", "개념"],
                    "전문용어": ["살아있는", "자연인", "식별가능", "정보주체"],
                    "선호답변": {"1": 0.28, "2": 0.25, "3": 0.22, "4": 0.15, "5": 0.10},
                    "신뢰도": 0.74,
                    "특별규칙": "개인정보보호법_참조패턴"
                }
            },
            "문맥_분석패턴": {
                "정의형_질문": {
                    "지시어": ["정의", "의미", "개념", "뜻"],
                    "답변구조": "정의_제시_후_세부설명",
                    "신뢰도_가산": 0.1
                },
                "절차형_질문": {
                    "지시어": ["절차", "과정", "단계", "방법"],
                    "답변구조": "단계별_순차설명",
                    "신뢰도_가산": 0.08
                },
                "비교형_질문": {
                    "지시어": ["차이", "비교", "구분", "대비"],
                    "답변구조": "비교대상_명시_후_차이점설명",
                    "신뢰도_가산": 0.12
                }
            }
        }
    
    def _build_multi_level_patterns(self) -> Dict:
        return {
            "레벨1_단순패턴": {
                "키워드_일치": {"가중치": 0.3, "최소_매치": 1},
                "도메인_식별": {"가중치": 0.4, "신뢰도_임계": 0.3}
            },
            "레벨2_구조패턴": {
                "문장구조_분석": {"가중치": 0.5, "최소_매치": 2},
                "의미관계_파악": {"가중치": 0.6, "신뢰도_임계": 0.5}
            },
            "레벨3_의미패턴": {
                "의도파악": {"가중치": 0.8, "최소_매치": 3},
                "논리구조_분석": {"가중치": 0.9, "신뢰도_임계": 0.7}
            }
        }
    
    def analyze_patterns(self, question: str, question_structure: Dict) -> Dict:
        self.total_analyses += 1
        
        cache_key = hashlib.md5(question.encode()).hexdigest()[:12]
        if cache_key in self.semantic_similarity_cache:
            self.cache_hit_rate = (self.cache_hit_rate * (self.total_analyses - 1) + 1) / self.total_analyses
            return self.semantic_similarity_cache[cache_key]
        
        analysis_result = {
            "패턴레벨": 1,
            "도메인_신뢰도": {},
            "의미_벡터": [],
            "문맥_강도": 0.0,
            "추천_답변": {},
            "분석_깊이": "기본",
            "특별_규칙": [],
            "cross_domain_signals": []
        }
        
        question_normalized = re.sub(r'\s+', '', question.lower())
        
        domain_scores = self._analyze_domain_with_weights(question_normalized)
        analysis_result["도메인_신뢰도"] = domain_scores
        
        pattern_level = self._determine_pattern_level(question, domain_scores)
        analysis_result["패턴레벨"] = pattern_level
        
        if pattern_level >= 2:
            structural_analysis = self._analyze_question_structure(question, question_structure)
            analysis_result.update(structural_analysis)
        
        if pattern_level >= 3:
            semantic_analysis = self._analyze_semantic_patterns(question_normalized, domain_scores)
            analysis_result.update(semantic_analysis)
        
        ngram_patterns = self._extract_ngram_patterns(question_normalized)
        analysis_result["ngram_patterns"] = ngram_patterns
        
        special_rules = self._apply_special_rules(question, domain_scores)
        analysis_result["특별_규칙"] = special_rules
        
        cross_domain = self._analyze_cross_domain_patterns(question_normalized, domain_scores)
        analysis_result["cross_domain_signals"] = cross_domain
        
        recommended_answer = self._generate_pattern_based_answer(analysis_result)
        analysis_result["추천_답변"] = recommended_answer
        
        self._update_cache(cache_key, analysis_result)
        
        return analysis_result
    
    def _analyze_domain_with_weights(self, question: str) -> Dict[str, float]:
        domain_scores = {}
        
        for domain, domain_data in self.domain_keywords.items():
            total_score = 0.0
            match_count = 0
            
            weights = domain_data["가중치"]
            
            for category, keywords in domain_data.items():
                if category == "가중치":
                    continue
                
                category_weight = weights.get(category.split("키워드")[0], 0.5)
                
                for keyword in keywords:
                    if keyword in question:
                        total_score += category_weight
                        match_count += 1
            
            if match_count > 0:
                domain_scores[domain] = min(total_score / max(len([k for cat, k_list in domain_data.items() if cat != "가중치" for k in k_list]), 1), 1.0)
        
        return domain_scores
    
    def _determine_pattern_level(self, question: str, domain_scores: Dict) -> int:
        question_lower = question.lower()
        
        max_domain_score = max(domain_scores.values()) if domain_scores else 0
        
        complexity_indicators = 0
        
        if max_domain_score > 0.6:
            complexity_indicators += 1
        
        if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
            complexity_indicators += 1
        
        if len(re.findall(r'[.!?]', question)) > 1:
            complexity_indicators += 1
        
        if len(question) > 150:
            complexity_indicators += 1
        
        technical_terms = len(re.findall(r'(?:법|조|항|호|규정|정책|체계|절차)', question))
        if technical_terms >= 3:
            complexity_indicators += 1
        
        if complexity_indicators >= 4:
            return 3
        elif complexity_indicators >= 2:
            return 2
        else:
            return 1
    
    def _analyze_question_structure(self, question: str, structure: Dict) -> Dict:
        result = {
            "구조_분석": "완료",
            "문맥_강도": 0.0,
            "의미_연결성": 0.0
        }
        
        question_lower = question.lower()
        
        context_strength = 0.0
        
        if structure.get("has_negative", False):
            context_strength += 0.3
        
        choice_count = structure.get("choice_count", 0)
        if choice_count >= 4:
            context_strength += 0.2
        
        technical_terms = len(structure.get("technical_terms", []))
        context_strength += min(technical_terms * 0.1, 0.3)
        
        legal_refs = len(structure.get("legal_references", []))
        context_strength += min(legal_refs * 0.15, 0.2)
        
        result["문맥_강도"] = min(context_strength, 1.0)
        
        semantic_connectivity = 0.0
        for rule_type, rules in self.pattern_rules["문맥_분석패턴"].items():
            indicators = rules.get("지시어", [])
            if any(indicator in question_lower for indicator in indicators):
                semantic_connectivity += 0.3
                result["분석_깊이"] = f"{rule_type}_감지"
        
        result["의미_연결성"] = min(semantic_connectivity, 1.0)
        
        return result
    
    def _analyze_semantic_patterns(self, question: str, domain_scores: Dict) -> Dict:
        result = {
            "의미_벡터": [],
            "패턴_클러스터": None,
            "유사도_점수": 0.0
        }
        
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "일반"
        
        if primary_domain in self.domain_keywords:
            domain_keywords = []
            for category, keywords in self.domain_keywords[primary_domain].items():
                if category != "가중치":
                    domain_keywords.extend(keywords)
            
            semantic_vector = []
            for keyword in domain_keywords[:20]:
                if keyword in question:
                    semantic_vector.append(1.0)
                else:
                    similarity = self._calculate_keyword_similarity(keyword, question)
                    semantic_vector.append(similarity)
            
            result["의미_벡터"] = semantic_vector[:10]
            result["유사도_점수"] = np.mean(semantic_vector) if semantic_vector else 0.0
        
        return result
    
    def _calculate_keyword_similarity(self, keyword: str, question: str) -> float:
        keyword_chars = set(keyword)
        question_chars = set(question)
        
        intersection = len(keyword_chars & question_chars)
        union = len(keyword_chars | question_chars)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        if keyword in question:
            return 1.0
        elif any(char_group in question for char_group in [keyword[:len(keyword)//2], keyword[len(keyword)//2:]]):
            return 0.6 + jaccard_similarity * 0.4
        else:
            return jaccard_similarity * 0.3
    
    def _extract_ngram_patterns(self, question: str) -> List[str]:
        words = re.findall(r'[가-힣]{2,}', question)
        
        patterns = []
        
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            patterns.append(bigram)
            self.ngram_patterns[2][bigram] += 1
        
        for i in range(len(words) - 2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            patterns.append(trigram)
            self.ngram_patterns[3][trigram] += 1
        
        return patterns[:5]
    
    def _apply_special_rules(self, question: str, domain_scores: Dict) -> List[str]:
        question_lower = question.lower()
        applied_rules = []
        
        for rule_category, rules in self.pattern_rules["부정표현_패턴"].items():
            keywords = rules["키워드조합"]
            if all(keyword in question_lower for keyword in keywords):
                applied_rules.append(f"부정표현_{rule_category}")
        
        for rule_category, rules in self.pattern_rules["도메인_특화패턴"].items():
            keywords = rules["키워드조합"]
            if all(keyword in question_lower for keyword in keywords):
                applied_rules.append(f"도메인특화_{rule_category}")
        
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None
        if primary_domain:
            applied_rules.append(f"주도메인_{primary_domain}")
        
        return applied_rules
    
    def _analyze_cross_domain_patterns(self, question: str, domain_scores: Dict) -> List[str]:
        cross_signals = []
        
        active_domains = [domain for domain, score in domain_scores.items() if score > 0.3]
        
        if len(active_domains) >= 2:
            cross_signals.append("다중도메인_감지")
            
            for i, domain1 in enumerate(active_domains):
                for domain2 in active_domains[i+1:]:
                    transition_key = f"{domain1}→{domain2}"
                    self.domain_transition_matrix[domain1][domain2] += 1
                    cross_signals.append(f"도메인전환_{transition_key}")
        
        legal_terms = len(re.findall(r'(?:법|조|항|규정|정책)', question))
        technical_terms = len(re.findall(r'(?:시스템|기술|암호|보안)', question))
        
        if legal_terms >= 2 and technical_terms >= 2:
            cross_signals.append("법기술_융합패턴")
        
        return cross_signals
    
    def _generate_pattern_based_answer(self, analysis: Dict) -> Dict:
        result = {
            "추천답변": None,
            "신뢰도": 0.0,
            "근거": []
        }
        
        special_rules = analysis.get("특별_규칙", [])
        domain_scores = analysis.get("도메인_신뢰도", {})
        pattern_level = analysis.get("패턴레벨", 1)
        
        base_confidence = 0.4
        
        for rule in special_rules:
            if "부정표현" in rule:
                for rule_name, rule_data in self.pattern_rules["부정표현_패턴"].items():
                    if rule_name in rule:
                        preferred_answers = rule_data["선호답변"]
                        confidence = rule_data["신뢰도"]
                        
                        if confidence > base_confidence:
                            result["추천답변"] = preferred_answers
                            result["신뢰도"] = confidence
                            result["근거"].append(f"부정표현패턴_{rule_name}")
                            base_confidence = confidence
            
            elif "도메인특화" in rule:
                for rule_name, rule_data in self.pattern_rules["도메인_특화패턴"].items():
                    if rule_name in rule:
                        preferred_answers = rule_data["선호답변"]
                        confidence = rule_data["신뢰도"]
                        
                        if confidence > base_confidence:
                            result["추천답변"] = preferred_answers
                            result["신뢰도"] = confidence
                            result["근거"].append(f"도메인특화패턴_{rule_name}")
                            base_confidence = confidence
        
        pattern_bonus = (pattern_level - 1) * 0.05
        result["신뢰도"] = min(result["신뢰도"] + pattern_bonus, 0.85)
        
        if len(domain_scores) > 0:
            max_domain_score = max(domain_scores.values())
            domain_bonus = max_domain_score * 0.1
            result["신뢰도"] = min(result["신뢰도"] + domain_bonus, 0.90)
        
        cross_domain_signals = analysis.get("cross_domain_signals", [])
        if len(cross_domain_signals) >= 2:
            result["신뢰도"] *= 0.9
            result["근거"].append("다중도메인_복잡성_보정")
        
        return result
    
    def _update_cache(self, cache_key: str, analysis: Dict):
        if len(self.semantic_similarity_cache) >= 200:
            oldest_key = next(iter(self.semantic_similarity_cache))
            del self.semantic_similarity_cache[oldest_key]
        
        self.semantic_similarity_cache[cache_key] = analysis
    
    def learn_from_success(self, question: str, correct_answer: str, confidence: float):
        if confidence < 0.5:
            return
        
        self.successful_predictions += 1
        
        question_normalized = re.sub(r'\s+', '', question.lower())
        ngram_patterns = self._extract_ngram_patterns(question_normalized)
        
        for pattern in ngram_patterns:
            self.pattern_success_history[pattern].append({
                "answer": correct_answer,
                "confidence": confidence,
                "timestamp": time.time()
            })
            
            if len(self.pattern_success_history[pattern]) > 20:
                self.pattern_success_history[pattern] = self.pattern_success_history[pattern][-20:]
        
        domain_scores = self._analyze_domain_with_weights(question_normalized)
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "일반"
        
        evolution_data = {
            "domain": primary_domain,
            "confidence": confidence,
            "patterns": ngram_patterns[:3],
            "timestamp": time.time()
        }
        
        self.pattern_evolution_tracker[primary_domain].append(evolution_data)
        if len(self.pattern_evolution_tracker[primary_domain]) > 50:
            self.pattern_evolution_tracker[primary_domain] = self.pattern_evolution_tracker[primary_domain][-50:]
    
    def get_prediction(self, question: str, question_structure: Dict) -> Tuple[Optional[str], float]:
        analysis = self.analyze_patterns(question, question_structure)
        
        pattern_answer = analysis.get("추천_답변", {})
        
        if pattern_answer.get("추천답변") and pattern_answer.get("신뢰도", 0) > 0.45:
            answer_distribution = pattern_answer["추천답변"]
            
            if isinstance(answer_distribution, dict):
                sorted_answers = sorted(answer_distribution.items(), key=lambda x: x[1], reverse=True)
                if sorted_answers:
                    best_answer = sorted_answers[0][0]
                    confidence = pattern_answer["신뢰도"]
                    
                    temporal_boost = self._calculate_temporal_boost(question)
                    final_confidence = min(confidence + temporal_boost, 0.88)
                    
                    return best_answer, final_confidence
        
        fallback_answer = self._get_fallback_prediction(question, analysis)
        return fallback_answer, 0.35
    
    def _calculate_temporal_boost(self, question: str) -> float:
        question_normalized = re.sub(r'\s+', '', question.lower())
        
        recent_patterns = []
        current_time = time.time()
        
        for pattern, history in self.pattern_success_history.items():
            if pattern in question_normalized:
                recent_successes = [h for h in history if current_time - h["timestamp"] < 3600]
                if recent_successes:
                    avg_confidence = sum(h["confidence"] for h in recent_successes) / len(recent_successes)
                    recent_patterns.append(avg_confidence)
        
        if recent_patterns:
            temporal_boost = np.mean(recent_patterns) * 0.1
            return min(temporal_boost, 0.15)
        
        return 0.0
    
    def _get_fallback_prediction(self, question: str, analysis: Dict) -> str:
        domain_scores = analysis.get("도메인_신뢰도", {})
        has_negative = "부정표현" in str(analysis.get("특별_규칙", []))
        
        if has_negative:
            return "3"
        
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            domain_preferences = {
                "개인정보보호": "1",
                "전자금융": "2", 
                "정보보안": "1",
                "사이버보안": "2",
                "위험관리": "3"
            }
            return domain_preferences.get(primary_domain, "1")
        
        return "1"
    
    def optimize_patterns(self) -> Dict:
        optimization_results = {
            "patterns_optimized": 0,
            "patterns_removed": 0,
            "confidence_improved": 0,
            "cache_efficiency": 0.0
        }
        
        current_time = time.time()
        patterns_to_remove = []
        
        for pattern, history in self.pattern_success_history.items():
            recent_history = [h for h in history if current_time - h["timestamp"] < 7200]
            
            if len(recent_history) < 2:
                patterns_to_remove.append(pattern)
            else:
                avg_confidence = sum(h["confidence"] for h in recent_history) / len(recent_history)
                if avg_confidence > 0.7:
                    self.temporal_pattern_weights[pattern] = min(self.temporal_pattern_weights[pattern] + 0.1, 1.5)
                    optimization_results["confidence_improved"] += 1
                else:
                    self.temporal_pattern_weights[pattern] = max(self.temporal_pattern_weights[pattern] - 0.05, 0.5)
                
                optimization_results["patterns_optimized"] += 1
        
        for pattern in patterns_to_remove:
            del self.pattern_success_history[pattern]
            if pattern in self.temporal_pattern_weights:
                del self.temporal_pattern_weights[pattern]
            optimization_results["patterns_removed"] += 1
        
        optimization_results["cache_efficiency"] = self.cache_hit_rate
        
        return optimization_results
    
    def get_performance_stats(self) -> Dict:
        success_rate = self.successful_predictions / max(self.total_analyses, 1)
        
        return {
            "total_analyses": self.total_analyses,
            "successful_predictions": self.successful_predictions,
            "success_rate": success_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "active_patterns": len(self.pattern_success_history),
            "domain_clusters": len(self.domain_keywords),
            "cross_domain_transitions": len(self.domain_transition_matrix),
            "semantic_cache_size": len(self.semantic_similarity_cache),
            "temporal_weights_active": len(self.temporal_pattern_weights)
        }
    
    def save_patterns(self, filepath: str = "./patterns.pkl") -> bool:
        try:
            pattern_data = {
                "ngram_patterns": dict(self.ngram_patterns),
                "pattern_success_history": dict(self.pattern_success_history),
                "domain_transition_matrix": dict(self.domain_transition_matrix),
                "temporal_pattern_weights": dict(self.temporal_pattern_weights),
                "pattern_evolution_tracker": dict(self.pattern_evolution_tracker),
                "performance_stats": self.get_performance_stats()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(pattern_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return True
        except Exception:
            return False
    
    def load_patterns(self, filepath: str = "./patterns.pkl") -> bool:
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'rb') as f:
                pattern_data = pickle.load(f)
            
            self.ngram_patterns = defaultdict(lambda: defaultdict(int), pattern_data.get("ngram_patterns", {}))
            self.pattern_success_history = defaultdict(list, pattern_data.get("pattern_success_history", {}))
            self.domain_transition_matrix = defaultdict(lambda: defaultdict(float), pattern_data.get("domain_transition_matrix", {}))
            self.temporal_pattern_weights = defaultdict(float, pattern_data.get("temporal_pattern_weights", {}))
            self.pattern_evolution_tracker = defaultdict(list, pattern_data.get("pattern_evolution_tracker", {}))
            
            stats = pattern_data.get("performance_stats", {})
            self.total_analyses = stats.get("total_analyses", 0)
            self.successful_predictions = stats.get("successful_predictions", 0)
            self.cache_hit_rate = stats.get("cache_hit_rate", 0.0)
            
            if self.debug_mode:
                print(f"패턴 데이터 로드: {len(self.pattern_success_history)}개 패턴")
            
            return True
        except Exception:
            return False
    
    def cleanup(self):
        pattern_count = len(self.pattern_success_history)
        cache_efficiency = self.cache_hit_rate
        
        self.semantic_similarity_cache.clear()
        self.pattern_success_history.clear()
        self.domain_transition_matrix.clear()
        
        if self.debug_mode:
            print(f"패턴 분석기 정리: {pattern_count}개 패턴, 캐시효율 {cache_efficiency:.2%}")