# pattern_learner.py
"""
답변 패턴 학습 및 예측 시스템
"""

import re
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

def _default_int():
    """기본 정수값 반환"""
    return 0

def _default_float():
    """기본 실수값 반환"""
    return 0.0

def _default_list():
    """기본 리스트 반환"""
    return []

def _default_counter():
    """기본 카운터 반환"""
    return Counter()

class AnswerPatternLearner:
    """답변 패턴 학습 클래스"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.patterns = {
            "keyword_answer_map": defaultdict(_default_counter),
            "domain_answer_distribution": defaultdict(_default_counter),
            "negative_answer_patterns": Counter(),
            "question_type_patterns": defaultdict(_default_counter)
        }
        
        self.learned_rules = self._initialize_rules()
        
        self.pattern_performance = {
            "rule_success_rate": defaultdict(_default_list),
            "prediction_accuracy": defaultdict(_default_float)
        }
        
        self.prediction_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 200
        
    def _debug_print(self, message: str):
        """디버그 출력 (조건부)"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
        
    def _initialize_rules(self) -> Dict:
        """핵심 규칙만 유지"""
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미", "개념"],
                "preferred_answers": {"2": 0.65, "1": 0.20, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적", "거래"],
                "preferred_answers": {"2": 0.60, "3": 0.25, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.70
            },
            "유출_신고": {
                "keywords": ["유출", "신고", "즉시", "지체", "통지"],
                "preferred_answers": {"1": 0.70, "2": 0.15, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.80
            },
            "부정형_일반": {
                "keywords": ["해당하지않는", "적절하지않은", "옳지않은", "틀린"],
                "preferred_answers": {"1": 0.40, "5": 0.30, "4": 0.20, "2": 0.06, "3": 0.04},
                "confidence": 0.65
            },
            "보안_조치": {
                "keywords": ["보안", "조치", "대책", "방안", "관리"],
                "preferred_answers": {"3": 0.45, "2": 0.30, "1": 0.15, "4": 0.06, "5": 0.04},
                "confidence": 0.65
            },
            "접근매체_관리": {
                "keywords": ["접근매체", "안전", "관리", "선정"],
                "preferred_answers": {"1": 0.60, "2": 0.25, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.70
            }
        }
    
    def analyze_question_pattern(self, question: str) -> Optional[Dict]:
        """간소화된 패턴 분석"""
        
        q_hash = hash(question[:100])
        if q_hash in self.pattern_cache:
            return self.pattern_cache[q_hash]
        
        question_lower = question.lower().replace(" ", "")
        
        best_rule = None
        best_score = 0
        
        for rule_name, rule_info in self.learned_rules.items():
            keywords = rule_info["keywords"]
            
            matches = sum(1 for kw in keywords if kw.replace(" ", "") in question_lower)
            
            if matches > 0:
                score = matches / len(keywords)
                
                if score > best_score:
                    best_score = score
                    best_rule = {
                        "rule": rule_name,
                        "match_score": score,
                        "base_confidence": rule_info["confidence"],
                        "answers": rule_info["preferred_answers"]
                    }
        
        if len(self.pattern_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[q_hash] = best_rule
        return best_rule
    
    def predict_answer(self, question: str, structure: Dict) -> Tuple[str, float]:
        """간소화된 답변 예측"""
        
        cache_key = hash(f"{question[:50]}{structure.get('question_type', '')}")
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if structure.get("has_negative", False):
            result = self._predict_negative_simple(question, structure)
        else:
            pattern_match = self.analyze_question_pattern(question)
            
            if pattern_match and pattern_match["base_confidence"] > 0.6:
                answers = pattern_match["answers"]
                best_answer = max(answers.items(), key=lambda x: x[1])
                confidence = pattern_match["base_confidence"] * pattern_match["match_score"]
                result = (best_answer[0], min(confidence, 0.9))
            else:
                result = self._statistical_prediction_simple(question, structure)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = result
        return result
    
    def _predict_negative_simple(self, question: str, structure: Dict) -> Tuple[str, float]:
        """간소화된 부정형 예측"""
        question_lower = question.lower()
        
        if "모든" in question_lower or "항상" in question_lower:
            return "1", 0.75
        elif "제외" in question_lower or "빼고" in question_lower:
            return "5", 0.70
        elif "예외" in question_lower:
            return "4", 0.65
        else:
            domains = structure.get("domain", [])
            if "개인정보보호" in domains:
                return "1", 0.60
            elif "전자금융" in domains:
                return "2", 0.60
            else:
                return "1", 0.55
    
    def _statistical_prediction_simple(self, question: str, structure: Dict) -> Tuple[str, float]:
        """간소화된 통계 예측"""
        
        if structure["question_type"] != "multiple_choice":
            return "", 0.15
        
        length = len(question)
        
        if length < 200:
            return "2", 0.40
        elif length < 400:
            return "3", 0.35
        else:
            return "3", 0.30
    
    def update_patterns(self, question: str, correct_answer: str, structure: Dict,
                       prediction_result: Optional[Tuple[str, float]] = None):
        """간소화된 패턴 업데이트"""
        
        keywords = re.findall(r'[가-힣]{2,}', question.lower())
        for keyword in keywords[:3]:
            self.patterns["keyword_answer_map"][keyword][correct_answer] += 1
        
        domains = structure.get("domain", ["일반"])
        for domain in domains:
            self.patterns["domain_answer_distribution"][domain][correct_answer] += 1
        
        if structure.get("has_negative", False):
            self.patterns["negative_answer_patterns"][correct_answer] += 1
        
        if prediction_result:
            predicted_answer, confidence = prediction_result
            is_correct = (predicted_answer == correct_answer)
            
            pattern_match = self.analyze_question_pattern(question)
            if pattern_match:
                rule_name = pattern_match["rule"]
                self.pattern_performance["rule_success_rate"][rule_name].append(is_correct)
    
    def get_confidence_boost(self, question: str, predicted_answer: str, structure: Dict) -> float:
        """간소화된 신뢰도 부스트"""
        
        boost = 0.0
        
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            answers = pattern_match["answers"]
            if predicted_answer in answers:
                preference_score = answers[predicted_answer]
                boost += preference_score * 0.1
        
        domains = structure.get("domain", [])
        if domains and len(domains) <= 2:
            boost += 0.05
        
        return min(boost, 0.2)
    
    def get_pattern_insights(self) -> Dict:
        """간소화된 인사이트"""
        insights = {
            "rule_performance": {},
            "domain_preferences": {},
            "negative_distribution": dict(self.patterns["negative_answer_patterns"])
        }
        
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 3:
                success_rate = sum(success_list) / len(success_list)
                insights["rule_performance"][rule_name] = {
                    "success_rate": success_rate,
                    "sample_size": len(success_list)
                }
        
        for domain, answer_dist in self.patterns["domain_answer_distribution"].items():
            if sum(answer_dist.values()) >= 3:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["domain_preferences"][domain] = preferences
        
        return insights
    
    def optimize_rules(self):
        """간소화된 규칙 최적화"""
        
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 5:
                success_rate = sum(success_list) / len(success_list)
                
                if success_rate < 0.3 and rule_name in self.learned_rules:
                    self.learned_rules[rule_name]["confidence"] *= 0.9
                elif success_rate > 0.8 and rule_name in self.learned_rules:
                    current_confidence = self.learned_rules[rule_name]["confidence"]
                    self.learned_rules[rule_name]["confidence"] = min(current_confidence * 1.02, 0.95)
    
    def save_patterns(self, filepath: str = "./learned_patterns.pkl"):
        """패턴 저장"""
        save_data = {
            "patterns": {k: dict(v) if hasattr(v, 'items') else v for k, v in self.patterns.items()},
            "rules": self.learned_rules,
            "performance": {k: dict(v) if hasattr(v, 'items') else v for k, v in self.pattern_performance.items()}
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def load_patterns(self, filepath: str = "./learned_patterns.pkl"):
        """패턴 로드"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                self.patterns = defaultdict(_default_counter, data.get("patterns", {}))
                self.learned_rules = data.get("rules", self.learned_rules)
                
                if "performance" in data:
                    perf_data = data["performance"]
                    self.pattern_performance = defaultdict(_default_list, perf_data)
                
                return True
        except Exception:
            return False
    
    def cleanup(self):
        """리소스 정리"""
        cache_size = len(self.prediction_cache) + len(self.pattern_cache)
        if cache_size > 0 and self.debug_mode:
            print(f"패턴 학습기 캐시: {cache_size}개")
        
        self.prediction_cache.clear()
        self.pattern_cache.clear()

class SmartAnswerSelector:
    """간소화된 답변 선택기"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.pattern_learner = AnswerPatternLearner(debug_mode=debug_mode)
        self.selection_stats = {
            "total_selections": 0,
            "pattern_based": 0,
            "model_based": 0
        }
        
    def _debug_print(self, message: str):
        """디버그 출력 (조건부)"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def select_best_answer(self, question: str, model_response: str, 
                         structure: Dict, confidence: float) -> Tuple[str, float]:
        """간소화된 답변 선택"""
        
        self.selection_stats["total_selections"] += 1
        
        extracted_answers = self._extract_answers_fast(model_response)
        
        if extracted_answers:
            answer = extracted_answers[0]
            
            final_confidence = min(
                confidence + self.pattern_learner.get_confidence_boost(question, answer, structure),
                1.0
            )
            
            self.selection_stats["model_based"] += 1
            return answer, final_confidence
        
        pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
        self.selection_stats["pattern_based"] += 1
        return pattern_answer, pattern_conf
    
    def _extract_answers_fast(self, response: str) -> List[str]:
        """빠른 답변 추출"""
        
        priority_patterns = [
            r'정답[:\s]*([1-5])',
            r'최종\s*답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'([1-5])번'
        ]
        
        for pattern in priority_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches
        
        numbers = re.findall(r'[1-5]', response)
        return numbers[:3] if numbers else []
    
    def get_selection_report(self) -> Dict:
        """선택 통계"""
        total = self.selection_stats["total_selections"]
        
        if total == 0:
            return {"message": "기록 없음"}
        
        return {
            "total_selections": total,
            "model_based_rate": self.selection_stats["model_based"] / total,
            "pattern_based_rate": self.selection_stats["pattern_based"] / total
        }
    
    def cleanup(self):
        """정리"""
        total = self.selection_stats["total_selections"]
        if total > 0 and self.debug_mode:
            print(f"답변 선택기: {total}회 선택")
        
        self.pattern_learner.cleanup()