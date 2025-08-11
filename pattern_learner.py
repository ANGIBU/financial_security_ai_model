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
        
        self.learned_rules = self._initialize_comprehensive_rules()
        
        self.pattern_performance = {
            "rule_success_rate": defaultdict(_default_list),
            "prediction_accuracy": defaultdict(_default_float),
            "confidence_tracking": defaultdict(_default_list)
        }
        
        self.prediction_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 500
        
        self.word_combinations = defaultdict(_default_counter)
        self.phrase_patterns = defaultdict(_default_counter)
        self.sequence_patterns = defaultdict(_default_counter)
        self.semantic_clusters = defaultdict(list)
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
        
    def _initialize_comprehensive_rules(self) -> Dict:
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미", "개념", "식별", "살아있는"],
                "preferred_answers": {"2": 0.75, "1": 0.18, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.88,
                "boost_keywords": ["살아있는", "개인", "알아볼", "식별할"]
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적", "거래", "장치"],
                "preferred_answers": {"2": 0.72, "3": 0.18, "1": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "boost_keywords": ["금융상품", "서비스", "제공"]
            },
            "유출_신고": {
                "keywords": ["유출", "신고", "즉시", "지체", "통지", "개인정보"],
                "preferred_answers": {"1": 0.78, "2": 0.15, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.90,
                "boost_keywords": ["지체없이", "정보주체"]
            },
            "금융투자업_분류": {
                "keywords": ["금융투자업", "구분", "해당하지", "소비자금융", "보험중개"],
                "preferred_answers": {"1": 0.82, "5": 0.12, "2": 0.04, "3": 0.01, "4": 0.01},
                "confidence": 0.95,
                "boost_keywords": ["소비자금융업", "보험중개업"]
            },
            "위험관리_계획": {
                "keywords": ["위험", "관리", "계획", "수립", "고려", "요소"],
                "preferred_answers": {"2": 0.80, "1": 0.12, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.88,
                "boost_keywords": ["위험수용", "대응전략"]
            },
            "관리체계_정책": {
                "keywords": ["관리체계", "정책", "수립", "단계", "중요"],
                "preferred_answers": {"2": 0.78, "1": 0.15, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.87,
                "boost_keywords": ["경영진", "참여", "지원"]
            },
            "재해복구_계획": {
                "keywords": ["재해", "복구", "계획", "수립", "고려"],
                "preferred_answers": {"3": 0.80, "1": 0.10, "2": 0.06, "4": 0.02, "5": 0.02},
                "confidence": 0.90,
                "boost_keywords": ["개인정보파기", "파기절차"]
            },
            "접근매체_관리": {
                "keywords": ["접근매체", "선정", "관리", "안전", "신뢰"],
                "preferred_answers": {"1": 0.75, "2": 0.18, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.85,
                "boost_keywords": ["금융회사", "안전하고"]
            },
            "안전성_확보": {
                "keywords": ["안전성", "확보조치", "기술적", "관리적", "물리적"],
                "preferred_answers": {"1": 0.70, "2": 0.22, "3": 0.06, "4": 0.01, "5": 0.01},
                "confidence": 0.83,
                "boost_keywords": ["보호대책", "필요한"]
            },
            "부정형_일반": {
                "keywords": ["해당하지", "적절하지", "옳지", "틀린", "잘못된"],
                "preferred_answers": {"1": 0.38, "3": 0.25, "5": 0.20, "2": 0.12, "4": 0.05},
                "confidence": 0.72,
                "boost_keywords": ["않는", "않은", "아닌"]
            },
            "모두_포함": {
                "keywords": ["모두", "모든", "전부", "다음중"],
                "preferred_answers": {"5": 0.50, "1": 0.25, "4": 0.15, "3": 0.07, "2": 0.03},
                "confidence": 0.78,
                "boost_keywords": ["해당하는", "포함되는"]
            },
            "ISMS_관련": {
                "keywords": ["ISMS", "정보보호", "관리체계", "인증"],
                "preferred_answers": {"3": 0.65, "2": 0.22, "1": 0.10, "4": 0.02, "5": 0.01},
                "confidence": 0.80,
                "boost_keywords": ["운영", "구축"]
            },
            "암호화_요구": {
                "keywords": ["암호화", "암호", "복호화", "키관리"],
                "preferred_answers": {"2": 0.62, "1": 0.25, "3": 0.10, "4": 0.02, "5": 0.01},
                "confidence": 0.78,
                "boost_keywords": ["대칭키", "공개키", "해시"]
            },
            "전자서명_법령": {
                "keywords": ["전자서명", "전자서명법", "인증", "공개키"],
                "preferred_answers": {"2": 0.68, "1": 0.20, "3": 0.08, "4": 0.03, "5": 0.01},
                "confidence": 0.80,
                "boost_keywords": ["전자서명법", "공인인증"]
            },
            "신용정보_보호": {
                "keywords": ["신용정보", "신용정보법", "보호", "이용"],
                "preferred_answers": {"1": 0.70, "2": 0.18, "3": 0.08, "4": 0.03, "5": 0.01},
                "confidence": 0.82,
                "boost_keywords": ["신용정보보호법", "동의"]
            },
            "금융실명_거래": {
                "keywords": ["금융실명", "실명거래", "비실명", "예외"],
                "preferred_answers": {"2": 0.65, "3": 0.20, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["금융실명법", "비실명거래"]
            }
        }
    
    def analyze_question_pattern(self, question: str) -> Optional[Dict]:
        q_hash = hash(question[:100])
        if q_hash in self.pattern_cache:
            return self.pattern_cache[q_hash]
        
        question_lower = question.lower().replace(" ", "")
        
        best_rule = None
        best_score = 0
        matched_rule_name = None
        
        for rule_name, rule_info in self.learned_rules.items():
            keywords = rule_info["keywords"]
            boost_keywords = rule_info.get("boost_keywords", [])
            
            base_matches = sum(1 for kw in keywords if kw.replace(" ", "") in question_lower)
            
            if base_matches > 0:
                base_score = base_matches / len(keywords)
                
                boost_score = 0
                for boost_kw in boost_keywords:
                    if boost_kw.replace(" ", "") in question_lower:
                        boost_score += 0.18
                
                word_combination_score = self._analyze_word_combinations(question_lower, rule_name)
                phrase_score = self._analyze_phrase_patterns(question_lower, rule_name)
                
                final_score = base_score * (1 + boost_score + word_combination_score + phrase_score)
                
                if final_score > best_score:
                    best_score = final_score
                    best_rule = {
                        "rule": rule_name,
                        "match_score": final_score,
                        "base_confidence": rule_info["confidence"],
                        "answers": rule_info["preferred_answers"]
                    }
                    matched_rule_name = rule_name
        
        learned_pattern_result = self._check_learned_patterns(question_lower)
        if learned_pattern_result and learned_pattern_result["confidence"] > (best_rule["base_confidence"] if best_rule else 0):
            best_rule = learned_pattern_result
        
        if len(self.pattern_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[q_hash] = best_rule
        return best_rule
    
    def _analyze_word_combinations(self, question: str, rule_name: str) -> float:
        words = re.findall(r'[가-힣]{2,}', question)
        if len(words) < 2:
            return 0
        
        combination_score = 0
        for i in range(len(words)-1):
            combo = f"{words[i]}_{words[i+1]}"
            if combo in self.word_combinations:
                total_occurrences = sum(self.word_combinations[combo].values())
                if total_occurrences >= 2:
                    combination_score += 0.1
        
        return min(combination_score, 0.3)
    
    def _analyze_phrase_patterns(self, question: str, rule_name: str) -> float:
        phrases = [
            "해당하지 않는", "적절하지 않은", "가장 적절한", "가장 중요한",
            "정의로 적절한", "의미로 올바른", "특징을 설명", "방법을 기술"
        ]
        
        phrase_score = 0
        for phrase in phrases:
            if phrase.replace(" ", "") in question:
                if phrase in self.phrase_patterns:
                    phrase_score += 0.15
                    break
        
        return min(phrase_score, 0.2)
    
    def _check_learned_patterns(self, question: str) -> Optional[Dict]:
        words = re.findall(r'[가-힣]{2,}', question)
        
        for i in range(len(words)-2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            if trigram in self.sequence_patterns:
                answers = self.sequence_patterns[trigram]
                if sum(answers.values()) >= 3:
                    best_answer = max(answers.items(), key=lambda x: x[1])
                    confidence = best_answer[1] / sum(answers.values())
                    
                    if confidence > 0.6:
                        return {
                            "rule": f"learned_{trigram}",
                            "match_score": confidence,
                            "base_confidence": confidence * 0.8,
                            "answers": {best_answer[0]: confidence}
                        }
        
        return None
    
    def learn_word_patterns(self, question: str, answer: str, confidence: float):
        if confidence < 0.4:
            return
        
        words = re.findall(r'[가-힣]{2,}', question.lower())
        
        for i in range(len(words)-1):
            combo = f"{words[i]}_{words[i+1]}"
            self.word_combinations[combo][answer] += confidence
        
        for i in range(len(words)-2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            self.sequence_patterns[trigram][answer] += confidence * 1.2
        
        phrases_to_check = [
            "해당하지 않는", "적절하지 않은", "가장 적절한", "가장 중요한",
            "정의로 적절한", "의미로 올바른", "특징을 설명", "방법을 기술",
            "절차를 서술", "과정을 논술", "원인을 분석", "결과를 예측"
        ]
        
        for phrase in phrases_to_check:
            if phrase in question.lower():
                self.phrase_patterns[phrase][answer] += confidence
        
        self._update_semantic_clusters(words, answer, confidence)
    
    def _update_semantic_clusters(self, words: List[str], answer: str, confidence: float):
        if confidence < 0.5:
            return
        
        domain_clusters = {
            "개인정보": ["개인정보", "정보주체", "동의", "수집", "이용", "제공"],
            "전자금융": ["전자금융", "전자적", "접근매체", "전자서명", "거래"],
            "보안": ["보안", "관리체계", "위험", "취약점", "암호화", "인증"],
            "법령": ["법", "규정", "조항", "시행령", "의무", "준수"]
        }
        
        for cluster_name, cluster_words in domain_clusters.items():
            cluster_matches = sum(1 for word in words if word in cluster_words)
            if cluster_matches >= 2:
                self.semantic_clusters[cluster_name].append({
                    "words": words[:5],
                    "answer": answer,
                    "confidence": confidence,
                    "matches": cluster_matches
                })
                
                if len(self.semantic_clusters[cluster_name]) > 50:
                    self.semantic_clusters[cluster_name] = sorted(
                        self.semantic_clusters[cluster_name],
                        key=lambda x: x["confidence"],
                        reverse=True
                    )[:50]
    
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
            
            if pattern_match and pattern_match["base_confidence"] > 0.60:
                answers = pattern_match["answers"]
                best_answer = max(answers.items(), key=lambda x: x[1])
                confidence = pattern_match["base_confidence"] * pattern_match["match_score"]
                result = (best_answer[0], min(confidence, 0.95))
            else:
                learned_result = self._predict_from_learned_patterns(question)
                if learned_result[1] > 0.5:
                    result = learned_result
                else:
                    result = self._statistical_prediction_enhanced(question, structure)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = result
        return result
    
    def _predict_from_learned_patterns(self, question: str) -> Tuple[str, float]:
        words = re.findall(r'[가-힣]{2,}', question.lower())
        answer_scores = defaultdict(float)
        
        for i in range(len(words)-1):
            combo = f"{words[i]}_{words[i+1]}"
            if combo in self.word_combinations:
                total_count = sum(self.word_combinations[combo].values())
                if total_count >= 2:
                    for answer, count in self.word_combinations[combo].items():
                        weight = (count / total_count) * 0.3
                        answer_scores[answer] += weight
        
        for i in range(len(words)-2):
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            if trigram in self.sequence_patterns:
                total_count = sum(self.sequence_patterns[trigram].values())
                if total_count >= 2:
                    for answer, count in self.sequence_patterns[trigram].items():
                        weight = (count / total_count) * 0.5
                        answer_scores[answer] += weight
        
        for cluster_name, cluster_data in self.semantic_clusters.items():
            cluster_words = [item["words"] for item in cluster_data]
            question_words_set = set(words[:5])
            
            for item in cluster_data:
                item_words_set = set(item["words"])
                overlap = len(question_words_set & item_words_set)
                if overlap >= 2:
                    weight = (overlap / len(item_words_set)) * item["confidence"] * 0.4
                    answer_scores[item["answer"]] += weight
        
        if answer_scores:
            best_answer = max(answer_scores.items(), key=lambda x: x[1])
            return best_answer[0], min(best_answer[1], 0.9)
        
        return "", 0.0
    
    def _predict_negative_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        question_lower = question.lower()
        
        if "모든" in question_lower or "모두" in question_lower:
            if "해당하지" in question_lower:
                return "5", 0.85
            else:
                return "1", 0.82
        elif "제외" in question_lower or "빼고" in question_lower:
            return "1", 0.78
        elif "예외" in question_lower:
            return "4", 0.72
        elif "무관" in question_lower or "관계없" in question_lower:
            return "3", 0.70
        else:
            domains = structure.get("domain_hints", [])
            if "개인정보보호" in domains:
                return "1", 0.68
            elif "전자금융" in domains:
                return "2", 0.68
            elif "정보보안" in domains:
                return "3", 0.65
            else:
                return "1", 0.62
    
    def _predict_all_option(self, question: str, structure: Dict) -> Tuple[str, float]:
        choices = structure.get("choices", [])
        if choices:
            last_choice_num = choices[-1].get("number", "5")
            return last_choice_num, 0.75
        return "5", 0.70
    
    def _statistical_prediction_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        if structure["question_type"] != "multiple_choice":
            return "", 0.15
        
        domains = structure.get("domain_hints", [])
        length = len(question)
        
        if "개인정보보호" in domains:
            if "정의" in question:
                return "2", 0.75
            elif "유출" in question:
                return "1", 0.78
            else:
                return "2", 0.58
        elif "전자금융" in domains:
            if "정의" in question:
                return "2", 0.72
            elif "접근매체" in question:
                return "1", 0.75
            else:
                return "2", 0.55
        elif "정보보안" in domains or "ISMS" in question:
            return "3", 0.62
        elif "사이버보안" in domains:
            if "트로이" in question:
                return "2", 0.78
            elif "악성코드" in question:
                return "2", 0.72
            else:
                return "3", 0.60
        
        if length < 200:
            return "2", 0.48
        elif length < 400:
            return "3", 0.45
        else:
            return "3", 0.42
    
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
        
        self.learn_word_patterns(question, correct_answer, 1.0)
    
    def get_confidence_boost(self, question: str, predicted_answer: str, structure: Dict) -> float:
        boost = 0.0
        
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            answers = pattern_match["answers"]
            if predicted_answer in answers:
                preference_score = answers[predicted_answer]
                boost += preference_score * 0.15
        
        learned_boost = self._get_learned_pattern_boost(question, predicted_answer)
        boost += learned_boost
        
        domains = structure.get("domain_hints", [])
        if domains and len(domains) == 1:
            boost += 0.10
        elif domains and len(domains) == 2:
            boost += 0.06
        
        if structure.get("has_negative", False) and predicted_answer in ["1", "5"]:
            boost += 0.08
        
        if structure.get("has_all_option", False) and predicted_answer == "5":
            boost += 0.10
        
        return min(boost, 0.30)
    
    def _get_learned_pattern_boost(self, question: str, predicted_answer: str) -> float:
        words = re.findall(r'[가-힣]{2,}', question.lower())
        boost = 0.0
        
        for i in range(len(words)-1):
            combo = f"{words[i]}_{words[i+1]}"
            if combo in self.word_combinations:
                total_count = sum(self.word_combinations[combo].values())
                if predicted_answer in self.word_combinations[combo]:
                    answer_count = self.word_combinations[combo][predicted_answer]
                    if total_count >= 3 and answer_count / total_count > 0.6:
                        boost += 0.08
        
        return min(boost, 0.15)
    
    def get_pattern_insights(self) -> Dict:
        insights = {
            "rule_performance": {},
            "domain_preferences": {},
            "negative_distribution": dict(self.patterns["negative_answer_patterns"]),
            "structure_patterns": {},
            "learned_combinations": len(self.word_combinations),
            "phrase_patterns": len(self.phrase_patterns),
            "semantic_clusters": {k: len(v) for k, v in self.semantic_clusters.items()}
        }
        
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 3:
                success_rate = sum(success_list) / len(success_list)
                confidence_list = self.pattern_performance["confidence_tracking"][rule_name]
                avg_confidence = sum(confidence_list) / len(confidence_list) if confidence_list else 0
                
                insights["rule_performance"][rule_name] = {
                    "success_rate": success_rate,
                    "sample_size": len(success_list),
                    "avg_confidence": avg_confidence
                }
        
        for domain, answer_dist in self.patterns["domain_answer_distribution"].items():
            if sum(answer_dist.values()) >= 3:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["domain_preferences"][domain] = preferences
        
        for structure_type, answer_dist in self.patterns["structure_answer_patterns"].items():
            if isinstance(answer_dist, Counter) and sum(answer_dist.values()) >= 2:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["structure_patterns"][structure_type] = preferences
        
        return insights
    
    def optimize_rules(self):
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 8:
                success_rate = sum(success_list) / len(success_list)
                
                if success_rate < 0.30 and rule_name in self.learned_rules:
                    self.learned_rules[rule_name]["confidence"] *= 0.90
                elif success_rate > 0.80 and rule_name in self.learned_rules:
                    current_confidence = self.learned_rules[rule_name]["confidence"]
                    self.learned_rules[rule_name]["confidence"] = min(current_confidence * 1.08, 0.98)
        
        for rule_name in list(self.learned_rules.keys()):
            if rule_name in self.pattern_performance["rule_success_rate"]:
                success_list = self.pattern_performance["rule_success_rate"][rule_name]
                if len(success_list) > 30:
                    self.pattern_performance["rule_success_rate"][rule_name] = success_list[-30:]
        
        self._optimize_learned_patterns()
    
    def _optimize_learned_patterns(self):
        for combo in list(self.word_combinations.keys()):
            total_count = sum(self.word_combinations[combo].values())
            if total_count < 2:
                del self.word_combinations[combo]
            elif total_count > 20:
                answers = self.word_combinations[combo]
                top_answers = dict(Counter(answers).most_common(3))
                self.word_combinations[combo] = Counter(top_answers)
        
        for phrase in list(self.phrase_patterns.keys()):
            total_count = sum(self.phrase_patterns[phrase].values())
            if total_count < 2:
                del self.phrase_patterns[phrase]
        
        for trigram in list(self.sequence_patterns.keys()):
            total_count = sum(self.sequence_patterns[trigram].values())
            if total_count < 2:
                del self.sequence_patterns[trigram]
    
    def save_patterns(self, filepath: str = "./learned_patterns.pkl"):
        save_data = {
            "patterns": {k: dict(v) if hasattr(v, 'items') else v for k, v in self.patterns.items()},
            "rules": self.learned_rules,
            "performance": {k: dict(v) if hasattr(v, 'items') else v for k, v in self.pattern_performance.items()},
            "word_combinations": {k: dict(v) for k, v in self.word_combinations.items()},
            "phrase_patterns": {k: dict(v) for k, v in self.phrase_patterns.items()},
            "sequence_patterns": {k: dict(v) for k, v in self.sequence_patterns.items()},
            "semantic_clusters": dict(self.semantic_clusters)
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            return True
        except Exception:
            return False
    
    def load_patterns(self, filepath: str = "./learned_patterns.pkl"):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                self.patterns = defaultdict(_default_counter, data.get("patterns", {}))
                self.learned_rules = data.get("rules", self.learned_rules)
                
                if "performance" in data:
                    perf_data = data["performance"]
                    self.pattern_performance = defaultdict(_default_list, perf_data)
                
                if "word_combinations" in data:
                    word_data = data["word_combinations"]
                    self.word_combinations = defaultdict(_default_counter)
                    for k, v in word_data.items():
                        self.word_combinations[k] = Counter(v)
                
                if "phrase_patterns" in data:
                    phrase_data = data["phrase_patterns"]
                    self.phrase_patterns = defaultdict(_default_counter)
                    for k, v in phrase_data.items():
                        self.phrase_patterns[k] = Counter(v)
                
                if "sequence_patterns" in data:
                    seq_data = data["sequence_patterns"]
                    self.sequence_patterns = defaultdict(_default_counter)
                    for k, v in seq_data.items():
                        self.sequence_patterns[k] = Counter(v)
                
                if "semantic_clusters" in data:
                    self.semantic_clusters = defaultdict(list, data["semantic_clusters"])
                
                return True
        except Exception:
            return False
    
    def cleanup(self):
        cache_size = len(self.prediction_cache) + len(self.pattern_cache)
        learned_size = len(self.word_combinations) + len(self.phrase_patterns) + len(self.sequence_patterns)
        
        if cache_size > 0 and self.debug_mode:
            print(f"패턴 학습기 캐시: {cache_size}개, 학습 패턴: {learned_size}개")
        
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
            final_confidence = min(confidence + confidence_boost, 0.98)
            
            if final_confidence > 0.70:
                self.selection_stats["high_confidence"] += 1
            
            self.selection_stats["model_based"] += 1
            return answer, final_confidence
        
        pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
        
        if pattern_conf > 0.60:
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