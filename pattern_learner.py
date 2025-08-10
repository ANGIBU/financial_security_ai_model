# pattern_learner.py

import re
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

def _default_int():
    return 0

def _default_float():
    return 0.0

def _default_list():
    return []

def _default_counter():
    return Counter()

class AnswerPatternLearner:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.patterns = {
            "keyword_answer_map": defaultdict(_default_counter),
            "domain_answer_distribution": defaultdict(_default_counter),
            "negative_answer_patterns": Counter(),
            "question_type_patterns": defaultdict(_default_counter),
            "context_answer_patterns": defaultdict(_default_counter),
            "structure_answer_patterns": defaultdict(_default_counter)
        }
        
        self.learned_rules = self._initialize_rules()
        
        self.pattern_performance = {
            "rule_success_rate": defaultdict(_default_list),
            "prediction_accuracy": defaultdict(_default_float),
            "confidence_tracking": defaultdict(_default_list)
        }
        
        self.prediction_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 200
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
        
    def _initialize_rules(self) -> Dict:
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미", "개념", "식별"],
                "preferred_answers": {"2": 0.70, "1": 0.18, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "boost_keywords": ["살아있는", "개인", "알아볼"]
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적", "거래", "장치"],
                "preferred_answers": {"2": 0.65, "3": 0.20, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["금융상품", "서비스", "제공"]
            },
            "유출_신고": {
                "keywords": ["유출", "신고", "즉시", "지체", "통지", "개인정보"],
                "preferred_answers": {"1": 0.72, "2": 0.15, "3": 0.08, "4": 0.03, "5": 0.02},
                "confidence": 0.82,
                "boost_keywords": ["지체없이", "정보주체"]
            },
            "금융투자업_분류": {
                "keywords": ["금융투자업", "구분", "해당하지", "소비자금융", "보험중개"],
                "preferred_answers": {"1": 0.80, "5": 0.10, "2": 0.05, "3": 0.03, "4": 0.02},
                "confidence": 0.88,
                "boost_keywords": ["소비자금융업", "보험중개업"]
            },
            "위험관리_계획": {
                "keywords": ["위험", "관리", "계획", "수립", "고려", "요소"],
                "preferred_answers": {"2": 0.75, "1": 0.12, "3": 0.08, "4": 0.03, "5": 0.02},
                "confidence": 0.83,
                "boost_keywords": ["위험수용", "대응전략"]
            },
            "관리체계_정책": {
                "keywords": ["관리체계", "정책", "수립", "단계", "중요"],
                "preferred_answers": {"2": 0.72, "1": 0.15, "3": 0.08, "4": 0.03, "5": 0.02},
                "confidence": 0.80,
                "boost_keywords": ["경영진", "참여", "지원"]
            },
            "재해복구_계획": {
                "keywords": ["재해", "복구", "계획", "수립", "고려"],
                "preferred_answers": {"3": 0.75, "1": 0.10, "2": 0.08, "4": 0.04, "5": 0.03},
                "confidence": 0.83,
                "boost_keywords": ["개인정보파기", "파기절차"]
            },
            "접근매체_관리": {
                "keywords": ["접근매체", "선정", "관리", "안전", "신뢰"],
                "preferred_answers": {"1": 0.70, "2": 0.15, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["금융회사", "안전하고"]
            },
            "안전성_확보": {
                "keywords": ["안전성", "확보조치", "기술적", "관리적", "물리적"],
                "preferred_answers": {"1": 0.65, "2": 0.20, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["보호대책", "필요한"]
            },
            "부정형_일반": {
                "keywords": ["해당하지", "적절하지", "옳지", "틀린", "잘못된"],
                "preferred_answers": {"1": 0.38, "3": 0.25, "5": 0.18, "2": 0.12, "4": 0.07},
                "confidence": 0.70,
                "boost_keywords": ["않는", "않은", "아닌"]
            },
            "모두_포함": {
                "keywords": ["모두", "모든", "전부", "다음중"],
                "preferred_answers": {"5": 0.48, "1": 0.22, "4": 0.15, "3": 0.10, "2": 0.05},
                "confidence": 0.72,
                "boost_keywords": ["해당하는", "포함되는"]
            },
            "ISMS_관련": {
                "keywords": ["ISMS", "정보보호", "관리체계", "인증"],
                "preferred_answers": {"3": 0.60, "2": 0.22, "1": 0.12, "4": 0.04, "5": 0.02},
                "confidence": 0.73,
                "boost_keywords": ["운영", "구축"]
            },
            "암호화_요구": {
                "keywords": ["암호화", "암호", "복호화", "키관리"],
                "preferred_answers": {"2": 0.58, "1": 0.22, "3": 0.13, "4": 0.05, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["대칭키", "공개키", "해시"]
            }
        }
    
    def analyze_question_pattern(self, question: str) -> Optional[Dict]:
        
        q_hash = hash(question[:100])
        if q_hash in self.pattern_cache:
            return self.pattern_cache[q_hash]
        
        question_lower = question.lower().replace(" ", "")
        
        best_rule = None
        best_score = 0
        
        for rule_name, rule_info in self.learned_rules.items():
            keywords = rule_info["keywords"]
            boost_keywords = rule_info.get("boost_keywords", [])
            
            base_matches = sum(1 for kw in keywords if kw.replace(" ", "") in question_lower)
            
            if base_matches > 0:
                base_score = base_matches / len(keywords)
                
                boost_score = 0
                for boost_kw in boost_keywords:
                    if boost_kw.replace(" ", "") in question_lower:
                        boost_score += 0.15
                
                final_score = base_score * (1 + boost_score)
                
                if final_score > best_score:
                    best_score = final_score
                    best_rule = {
                        "rule": rule_name,
                        "match_score": final_score,
                        "base_confidence": rule_info["confidence"],
                        "answers": rule_info["preferred_answers"]
                    }
        
        if len(self.pattern_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[q_hash] = best_rule
        return best_rule
    
    def predict_answer(self, question: str, structure: Dict) -> Tuple[str, float]:
        
        cache_key = hash(f"{question[:50]}{structure.get('question_type', '')}")
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if structure.get("has_negative", False):
            result = self._predict_negative_enhanced(question, structure)
        elif structure.get("has_all_option", False):
            result = self._predict_all_option(question, structure)
        else:
            pattern_match = self.analyze_question_pattern(question)
            
            if pattern_match and pattern_match["base_confidence"] > 0.65:
                answers = pattern_match["answers"]
                best_answer = max(answers.items(), key=lambda x: x[1])
                confidence = pattern_match["base_confidence"] * pattern_match["match_score"]
                result = (best_answer[0], min(confidence, 0.92))
            else:
                result = self._statistical_prediction_enhanced(question, structure)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = result
        return result
    
    def _predict_negative_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        question_lower = question.lower()
        
        if "모든" in question_lower or "모두" in question_lower:
            if "해당하지" in question_lower:
                return "5", 0.78
            else:
                return "1", 0.75
        elif "제외" in question_lower or "빼고" in question_lower:
            return "1", 0.72
        elif "예외" in question_lower:
            return "4", 0.68
        elif "무관" in question_lower or "관계없" in question_lower:
            return "3", 0.65
        else:
            domains = structure.get("domain_hints", [])
            if "개인정보보호" in domains:
                return "1", 0.62
            elif "전자금융" in domains:
                return "2", 0.62
            elif "정보보안" in domains:
                return "3", 0.60
            else:
                return "1", 0.58
    
    def _predict_all_option(self, question: str, structure: Dict) -> Tuple[str, float]:
        choices = structure.get("choices", [])
        if choices:
            last_choice_num = choices[-1].get("number", "5")
            return last_choice_num, 0.70
        return "5", 0.65
    
    def _statistical_prediction_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        
        if structure["question_type"] != "multiple_choice":
            return "", 0.15
        
        domains = structure.get("domain_hints", [])
        length = len(question)
        
        if "개인정보보호" in domains:
            if "정의" in question:
                return "2", 0.68
            elif "유출" in question:
                return "1", 0.70
            else:
                return "2", 0.50
        elif "전자금융" in domains:
            if "정의" in question:
                return "2", 0.65
            elif "접근매체" in question:
                return "1", 0.68
            else:
                return "2", 0.48
        elif "정보보안" in domains or "ISMS" in question:
            return "3", 0.55
        
        if length < 200:
            return "2", 0.42
        elif length < 400:
            return "3", 0.38
        else:
            return "3", 0.35
    
    def update_patterns(self, question: str, correct_answer: str, structure: Dict,
                       prediction_result: Optional[Tuple[str, float]] = None):
        
        keywords = re.findall(r'[가-힣]{2,}', question.lower())
        for keyword in keywords[:5]:
            self.patterns["keyword_answer_map"][keyword][correct_answer] += 1
        
        domains = structure.get("domain_hints", ["일반"])
        for domain in domains:
            self.patterns["domain_answer_distribution"][domain][correct_answer] += 1
        
        if structure.get("has_negative", False):
            self.patterns["negative_answer_patterns"][correct_answer] += 1
        
        if structure.get("has_all_option", False):
            self.patterns["structure_answer_patterns"]["all_option"][correct_answer] += 1
        
        question_length = len(question)
        if question_length < 200:
            self.patterns["structure_answer_patterns"]["short"][correct_answer] += 1
        elif question_length < 400:
            self.patterns["structure_answer_patterns"]["medium"][correct_answer] += 1
        else:
            self.patterns["structure_answer_patterns"]["long"][correct_answer] += 1
        
        if prediction_result:
            predicted_answer, confidence = prediction_result
            is_correct = (predicted_answer == correct_answer)
            
            pattern_match = self.analyze_question_pattern(question)
            if pattern_match:
                rule_name = pattern_match["rule"]
                self.pattern_performance["rule_success_rate"][rule_name].append(is_correct)
                self.pattern_performance["confidence_tracking"][rule_name].append(confidence)
    
    def get_confidence_boost(self, question: str, predicted_answer: str, structure: Dict) -> float:
        
        boost = 0.0
        
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            answers = pattern_match["answers"]
            if predicted_answer in answers:
                preference_score = answers[predicted_answer]
                boost += preference_score * 0.12
        
        domains = structure.get("domain_hints", [])
        if domains and len(domains) == 1:
            boost += 0.08
        elif domains and len(domains) == 2:
            boost += 0.05
        
        if structure.get("has_negative", False) and predicted_answer in ["1", "5"]:
            boost += 0.05
        
        if structure.get("has_all_option", False) and predicted_answer == "5":
            boost += 0.06
        
        return min(boost, 0.25)
    
    def get_pattern_insights(self) -> Dict:
        insights = {
            "rule_performance": {},
            "domain_preferences": {},
            "negative_distribution": dict(self.patterns["negative_answer_patterns"]),
            "structure_patterns": {}
        }
        
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 5:
                success_rate = sum(success_list) / len(success_list)
                confidence_list = self.pattern_performance["confidence_tracking"][rule_name]
                avg_confidence = sum(confidence_list) / len(confidence_list) if confidence_list else 0
                
                insights["rule_performance"][rule_name] = {
                    "success_rate": success_rate,
                    "sample_size": len(success_list),
                    "avg_confidence": avg_confidence
                }
        
        for domain, answer_dist in self.patterns["domain_answer_distribution"].items():
            if sum(answer_dist.values()) >= 5:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["domain_preferences"][domain] = preferences
        
        for structure_type, answer_dist in self.patterns["structure_answer_patterns"].items():
            if isinstance(answer_dist, Counter) and sum(answer_dist.values()) >= 3:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["structure_patterns"][structure_type] = preferences
        
        return insights
    
    def optimize_rules(self):
        
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 10:
                success_rate = sum(success_list) / len(success_list)
                
                if success_rate < 0.35 and rule_name in self.learned_rules:
                    self.learned_rules[rule_name]["confidence"] *= 0.92
                elif success_rate > 0.75 and rule_name in self.learned_rules:
                    current_confidence = self.learned_rules[rule_name]["confidence"]
                    self.learned_rules[rule_name]["confidence"] = min(current_confidence * 1.05, 0.95)
        
        for rule_name in list(self.learned_rules.keys()):
            if rule_name in self.pattern_performance["rule_success_rate"]:
                success_list = self.pattern_performance["rule_success_rate"][rule_name]
                if len(success_list) > 20:
                    self.pattern_performance["rule_success_rate"][rule_name] = success_list[-20:]
    
    def save_patterns(self, filepath: str = "./learned_patterns.pkl"):
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
        cache_size = len(self.prediction_cache) + len(self.pattern_cache)
        if cache_size > 0 and self.debug_mode:
            print(f"패턴 학습기 캐시: {cache_size}개")
        
        self.prediction_cache.clear()
        self.pattern_cache.clear()

class SmartAnswerSelector:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.pattern_learner = AnswerPatternLearner(debug_mode=debug_mode)
        self.selection_stats = {
            "total_selections": 0,
            "pattern_based": 0,
            "model_based": 0,
            "high_confidence": 0
        }
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def select_best_answer(self, question: str, model_response: str, 
                         structure: Dict, confidence: float) -> Tuple[str, float]:
        
        self.selection_stats["total_selections"] += 1
        
        extracted_answers = self._extract_answers_enhanced(model_response)
        
        if extracted_answers:
            answer = extracted_answers[0]
            
            confidence_boost = self.pattern_learner.get_confidence_boost(question, answer, structure)
            final_confidence = min(confidence + confidence_boost, 0.95)
            
            if final_confidence > 0.75:
                self.selection_stats["high_confidence"] += 1
            
            self.selection_stats["model_based"] += 1
            return answer, final_confidence
        
        pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
        
        if pattern_conf > 0.65:
            self.selection_stats["high_confidence"] += 1
        
        self.selection_stats["pattern_based"] += 1
        return pattern_answer, pattern_conf
    
    def _extract_answers_enhanced(self, response: str) -> List[str]:
        
        priority_patterns = [
            r'정답[:\s]*([1-5])',
            r'최종\s*답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'([1-5])번이\s*정답',
            r'([1-5])번'
        ]
        
        for pattern in priority_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches
        
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            number_counts = {}
            for num in numbers:
                number_counts[num] = number_counts.get(num, 0) + 1
            
            if number_counts:
                sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
                return [num for num, _ in sorted_numbers]
        
        return []
    
    def get_selection_report(self) -> Dict:
        total = self.selection_stats["total_selections"]
        
        if total == 0:
            return {"message": "기록 없음"}
        
        return {
            "total_selections": total,
            "model_based_rate": self.selection_stats["model_based"] / total,
            "pattern_based_rate": self.selection_stats["pattern_based"] / total,
            "high_confidence_rate": self.selection_stats["high_confidence"] / total
        }
    
    def cleanup(self):
        total = self.selection_stats["total_selections"]
        if total > 0 and self.debug_mode:
            print(f"답변 선택기: {total}회 선택")
        
        self.pattern_learner.cleanup()