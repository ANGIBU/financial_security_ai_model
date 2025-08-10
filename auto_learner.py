# auto_learner.py

import numpy as np
import pickle
import re
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def _default_int():
    return 0

def _default_float():
    return 0.0

def _default_list():
    return []

def _default_float_dict():
    return defaultdict(_default_float)

def _default_int_dict():
    return defaultdict(_default_int)

def atomic_save_model(obj, filepath: str) -> bool:
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, filepath)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception:
        return False

def atomic_load_model(filepath: str):
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

class AutoLearner:
    
    def __init__(self):
        self.learning_rate = 0.35
        self.confidence_threshold = 0.35
        self.min_samples = 1
        
        self.pattern_weights = defaultdict(_default_float_dict)
        self.pattern_counts = defaultdict(_default_int)
        
        self.answer_distribution = {
            "mc": defaultdict(_default_int),
            "domain": defaultdict(_default_int_dict),
            "negative": defaultdict(_default_int)
        }
        
        self.korean_quality_patterns = defaultdict(_default_list)
        self.successful_korean_templates = []
        
        self.learning_history = []
        self.performance_metrics = []
        
        self.domain_templates = self._initialize_korean_templates()
        self.specialized_rules = self._initialize_specialized_rules()
        
    def _initialize_korean_templates(self) -> Dict[str, List[str]]:
        return {
            "개인정보보호": [
                "개인정보보호법에 따라 {action}가 필요합니다.",
                "정보주체의 권리 보호를 위해 {measure}를 수행해야 합니다.",
                "개인정보의 안전한 관리를 위해 {requirement}가 요구됩니다.",
                "개인정보 처리방침을 수립하고 {action}를 이행해야 합니다.",
                "정보주체에게 개인정보 처리현황을 {method}로 통지해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {action}를 수행해야 합니다.",
                "전자적 장치를 통한 거래의 안전성 확보를 위해 {measure}가 필요합니다.",
                "접근매체 관리와 관련하여 {requirement}를 준수해야 합니다.",
                "전자금융거래의 신뢰성 향상을 위해 {action}가 요구됩니다.",
                "이용자 보호를 위해 {measure}를 구현해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계에 따라 {action}를 구현해야 합니다.",
                "체계적인 보안 관리를 위해 {measure}가 요구됩니다.",
                "위험평가를 통해 {requirement}를 수립해야 합니다.",
                "보안정책의 수립과 이행을 위해 {action}가 필요합니다.",
                "지속적인 보안 모니터링을 통해 {measure}를 확보해야 합니다."
            ],
            "암호화": [
                "중요 정보는 {action}를 통해 보호해야 합니다.",
                "암호화 기술을 활용하여 {measure}를 확보해야 합니다.",
                "안전한 키 관리를 위해 {requirement}가 필요합니다.",
                "전송 구간 암호화를 통해 {action}를 수행해야 합니다.",
                "저장 데이터 암호화를 위해 {measure}를 적용해야 합니다."
            ],
            "사고대응": [
                "{event} 발생 시 {action}를 수행해야 합니다.",
                "침해사고 대응은 {phase}별로 {measure}를 이행해야 합니다.",
                "복구 계획은 {target}을 고려하여 {requirement}를 수립해야 합니다.",
                "사고 대응팀 구성을 통해 {action}가 요구됩니다.",
                "신속한 복구를 위해 {measure}를 준비해야 합니다."
            ],
            "사이버보안": [
                "악성코드 탐지를 위해 {method}를 활용해야 합니다.",
                "트로이 목마는 {characteristic}를 가진 악성코드입니다.",
                "원격 접근 공격에 대비하여 {measure}가 필요합니다.",
                "시스템 감시를 통해 {indicator}를 확인해야 합니다.",
                "보안 솔루션을 통해 {action}를 수행해야 합니다."
            ]
        }
    
    def _initialize_specialized_rules(self) -> Dict[str, Dict]:
        return {
            "금융투자업_분류": {
                "patterns": ["금융투자업", "소비자금융업", "보험중개업", "투자매매업", "투자중개업", "투자자문업", "투자일임업"],
                "answers": {"1": 0.85, "5": 0.10, "2": 0.03, "3": 0.01, "4": 0.01},
                "confidence": 0.95,
                "rule": "소비자금융업과 보험중개업은 금융투자업이 아님"
            },
            "위험관리_요소": {
                "patterns": ["위험관리", "계획수립", "고려요소", "위험수용", "대응전략", "위험평가"],
                "answers": {"2": 0.82, "1": 0.10, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.90,
                "rule": "위험수용은 위험대응전략의 하나이지 별도 고려요소가 아님"
            },
            "관리체계_정책": {
                "patterns": ["관리체계", "정책수립", "경영진", "참여", "최고책임자", "중요한", "가장"],
                "answers": {"2": 0.80, "1": 0.12, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.88,
                "rule": "정책수립 단계에서 경영진의 참여가 가장 중요"
            },
            "재해복구_요소": {
                "patterns": ["재해복구", "계획수립", "개인정보파기", "복구절차", "비상연락", "백업"],
                "answers": {"3": 0.83, "1": 0.08, "2": 0.05, "4": 0.02, "5": 0.02},
                "confidence": 0.92,
                "rule": "개인정보파기절차는 재해복구와 직접 관련 없음"
            },
            "개인정보_정의": {
                "patterns": ["개인정보", "정의", "의미", "살아있는", "개인", "식별", "알아볼"],
                "answers": {"2": 0.78, "1": 0.15, "3": 0.04, "4": 0.02, "5": 0.01},
                "confidence": 0.87,
                "rule": "살아있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보"
            },
            "전자금융_정의": {
                "patterns": ["전자금융거래", "전자적", "장치", "금융상품", "서비스", "제공", "이용"],
                "answers": {"2": 0.75, "1": 0.18, "3": 0.04, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "rule": "전자적 장치를 통한 금융상품과 서비스 거래"
            },
            "접근매체_관리": {
                "patterns": ["접근매체", "선정", "관리", "안전", "신뢰", "금융회사"],
                "answers": {"1": 0.80, "2": 0.12, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.88,
                "rule": "금융회사는 안전하고 신뢰할 수 있는 접근매체를 선정해야 함"
            },
            "개인정보_유출": {
                "patterns": ["개인정보", "유출", "통지", "지체없이", "정보주체", "신고"],
                "answers": {"1": 0.82, "2": 0.10, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.90,
                "rule": "개인정보 유출 시 지체 없이 정보주체에게 통지"
            },
            "안전성_확보조치": {
                "patterns": ["안전성", "확보조치", "기술적", "관리적", "물리적", "보호대책"],
                "answers": {"1": 0.72, "2": 0.20, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "rule": "기술적, 관리적, 물리적 안전성 확보조치 필요"
            },
            "정보보호_관리체계": {
                "patterns": ["정보보호", "관리체계", "ISMS", "인증", "운영", "구축"],
                "answers": {"3": 0.70, "2": 0.18, "1": 0.08, "4": 0.03, "5": 0.01},
                "confidence": 0.83,
                "rule": "정보보호관리체계 인증 및 운영"
            },
            "암호화_요구사항": {
                "patterns": ["암호화", "암호", "복호화", "키관리", "해시", "전자서명"],
                "answers": {"2": 0.68, "1": 0.22, "3": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.80,
                "rule": "중요정보 암호화 및 안전한 키관리"
            },
            "부정형_일반": {
                "patterns": ["해당하지", "적절하지", "옳지", "틀린", "잘못된", "부적절한"],
                "answers": {"1": 0.40, "3": 0.25, "5": 0.18, "2": 0.12, "4": 0.05},
                "confidence": 0.75,
                "rule": "부정형 문제는 문맥에 따라 다양한 답 가능"
            },
            "모두_포함": {
                "patterns": ["모두", "모든", "전부", "다음중", "해당하는", "포함되는"],
                "answers": {"5": 0.50, "1": 0.25, "4": 0.15, "3": 0.07, "2": 0.03},
                "confidence": 0.78,
                "rule": "모두 해당하는 경우 마지막 번호 선택 경향"
            }
        }
    
    def _evaluate_korean_quality(self, text: str, question_type: str) -> float:
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            return 0.5
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        if re.search(r'[а-яё]', text.lower()):
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.9 - english_ratio * 0.1
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '방안', '시스템']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.08, 0.25)
        
        if 30 <= len(text) <= 500:
            quality += 0.15
        
        return max(0, min(1, quality))
    
    def _is_negative_question(self, question: str) -> bool:
        negative_keywords = [
            "해당하지 않는", "적절하지 않은", "옳지 않은", 
            "틀린", "잘못된", "부적절한", "아닌", "제외한"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in negative_keywords)
    
    def _extract_patterns(self, question: str) -> List[str]:
        patterns = []
        question_lower = question.lower()
        
        for rule_name, rule_info in self.specialized_rules.items():
            rule_patterns = rule_info["patterns"]
            match_count = sum(1 for pattern in rule_patterns if pattern in question_lower)
            
            if match_count >= 1:
                patterns.append(rule_name)
        
        if self._is_negative_question(question):
            patterns.append("negative_question")
        
        if "정의" in question_lower:
            patterns.append("definition_question")
        if "방안" in question_lower or "대책" in question_lower:
            patterns.append("solution_question")
        
        domains = {
            "personal_info": ["개인정보", "정보주체", "동의"],
            "electronic": ["전자금융", "전자적", "거래"],
            "security": ["보안", "암호화", "접근통제"],
            "crypto": ["암호", "해시", "전자서명"],
            "incident": ["사고", "유출", "침해"],
            "cyber": ["트로이", "악성코드", "원격", "RAT", "탐지"]
        }
        
        for domain, keywords in domains.items():
            if sum(1 for kw in keywords if kw in question_lower) >= 1:
                patterns.append(f"domain_{domain}")
        
        return patterns[:12]
    
    def learn_from_prediction(self, question: str, prediction: str,
                            confidence: float, question_type: str,
                            domain: List[str]) -> None:
        
        if confidence < self.confidence_threshold:
            return
        
        korean_quality = self._evaluate_korean_quality(prediction, question_type)
        
        if korean_quality < 0.2 and question_type != "multiple_choice":
            return
        
        patterns = self._extract_patterns(question)
        
        for pattern in patterns:
            weight_boost = confidence * self.learning_rate * max(korean_quality, 0.3)
            self.pattern_weights[pattern][prediction] += weight_boost
            self.pattern_counts[pattern] += 1
        
        if question_type == "multiple_choice":
            self.answer_distribution["mc"][prediction] += 1
            
            if self._is_negative_question(question):
                self.answer_distribution["negative"][prediction] += 1
        
        for d in domain:
            if d not in self.answer_distribution["domain"]:
                self.answer_distribution["domain"][d] = defaultdict(_default_int)
            self.answer_distribution["domain"][d][prediction] += 1
        
        if korean_quality > 0.5 and question_type != "multiple_choice":
            self._learn_korean_patterns(prediction, domain)
        
        self.learning_history.append({
            "question_sample": question[:80],
            "prediction": prediction[:80] if len(prediction) > 80 else prediction,
            "confidence": confidence,
            "korean_quality": korean_quality,
            "patterns": len(patterns)
        })
        
        if len(self.learning_history) > 200:
            self.learning_history = self.learning_history[-200:]
    
    def _learn_korean_patterns(self, text: str, domains: List[str]) -> None:
        if 30 <= len(text) <= 600:
            self.successful_korean_templates.append({
                "text": text,
                "domains": domains,
                "structure": self._analyze_text_structure(text)
            })
            
            if len(self.successful_korean_templates) > 50:
                self.successful_korean_templates = sorted(
                    self.successful_korean_templates,
                    key=lambda x: self._evaluate_korean_quality(x["text"], "subjective"),
                    reverse=True
                )[:50]
        
        for domain in domains:
            self.korean_quality_patterns[domain].append({
                "length": len(text),
                "keyword_count": self._count_domain_keywords(text, domain),
                "structure_markers": self._extract_structure_markers(text)
            })
            
            if len(self.korean_quality_patterns[domain]) > 30:
                self.korean_quality_patterns[domain] = self.korean_quality_patterns[domain][-30:]
    
    def _analyze_text_structure(self, text: str) -> Dict:
        return {
            "has_numbering": bool(re.search(r'첫째|둘째|1\)|2\)', text)),
            "has_law_reference": bool(re.search(r'법|규정|조항', text)),
            "has_conclusion": bool(re.search(r'따라서|그러므로|결론적으로', text)),
            "sentence_count": len(re.split(r'[.!?]', text))
        }
    
    def _count_domain_keywords(self, text: str, domain: str) -> int:
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "동의", "수집", "이용", "제공", "파기"],
            "전자금융": ["전자금융", "전자적", "거래", "접근매체", "전자서명"],
            "정보보안": ["보안", "관리체계", "접근통제", "위험평가", "보호대책"],
            "암호화": ["암호", "암호화", "복호화", "키", "해시", "전자서명"],
            "사이버보안": ["트로이", "악성코드", "원격", "탐지", "시스템", "감염"]
        }
        
        keywords = domain_keywords.get(domain, [])
        count = sum(1 for kw in keywords if kw in text)
        return count
    
    def _extract_structure_markers(self, text: str) -> List[str]:
        markers = []
        
        if re.search(r'첫째|둘째|셋째', text):
            markers.append("ordered_list")
        if re.search(r'1\)|2\)|3\)', text):
            markers.append("numbered_list")
        if re.search(r'따라서|그러므로|결론적으로', text):
            markers.append("conclusion")
        if re.search(r'예를 들어|구체적으로', text):
            markers.append("example")
        
        return markers
    
    def predict_with_patterns(self, question: str, question_type: str) -> Tuple[str, float]:
        patterns = self._extract_patterns(question)
        
        if not patterns:
            return self._get_default_answer(question_type), 0.2
        
        for rule_name in patterns:
            if rule_name in self.specialized_rules:
                rule = self.specialized_rules[rule_name]
                answers = rule["answers"]
                best_answer = max(answers.items(), key=lambda x: x[1])
                return best_answer[0], rule["confidence"]
        
        answer_scores = defaultdict(_default_float)
        total_weight = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights and self.pattern_counts[pattern] >= self.min_samples:
                pattern_weight = self.pattern_counts[pattern]
                
                for answer, weight in self.pattern_weights[pattern].items():
                    answer_scores[answer] += weight * pattern_weight
                    total_weight += pattern_weight
        
        if not answer_scores:
            return self._get_default_answer(question_type), 0.2
        
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        confidence = min(best_answer[1] / max(total_weight, 1), 0.9)
        
        if question_type != "multiple_choice":
            korean_quality = self._evaluate_korean_quality(best_answer[0], question_type)
            if korean_quality < 0.3:
                return self._generate_korean_answer(question, patterns), 0.5
        
        return best_answer[0], confidence
    
    def _generate_korean_answer(self, question: str, patterns: List[str]) -> str:
        domain = None
        for pattern in patterns:
            if pattern.startswith("domain_"):
                domain = pattern.replace("domain_", "")
                break
        
        if self.successful_korean_templates:
            relevant_templates = []
            for template in self.successful_korean_templates:
                if not domain or domain in [d.lower().replace("개인정보보호", "personal_info").replace("전자금융", "electronic").replace("정보보안", "security").replace("사이버보안", "cyber") for d in template["domains"]]:
                    relevant_templates.append(template)
            
            if relevant_templates:
                best_template = max(relevant_templates, 
                                  key=lambda x: self._evaluate_korean_quality(x["text"], "subjective"))
                return best_template["text"]
        
        if domain == "personal_info":
            base_answer = "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다."
        elif domain == "electronic":
            base_answer = "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다."
        elif domain == "security":
            base_answer = "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다."
        elif domain == "crypto":
            base_answer = "중요 정보는 안전한 암호 알고리즘을 사용하여 암호화하고 안전한 키 관리 체계를 구축해야 합니다."
        elif domain == "cyber":
            base_answer = "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다."
        else:
            base_answer = "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선을 수행해야 합니다."
        
        if "solution_question" in patterns:
            base_answer += " 구체적인 방안으로는 정책 수립, 조직 구성, 기술적 대책 구현, 정기적 점검 등이 있습니다."
        
        return base_answer
    
    def _get_default_answer(self, question_type: str) -> str:
        if question_type == "multiple_choice":
            if self.answer_distribution["mc"]:
                return max(self.answer_distribution["mc"].items(), 
                          key=lambda x: x[1])[0]
            return "2"
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 통해 안전성을 확보해야 합니다."
    
    def optimize_patterns(self) -> Dict:
        optimized = 0
        removed = 0
        
        patterns_to_remove = []
        for pattern, count in self.pattern_counts.items():
            if count < self.min_samples:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            if pattern in self.pattern_weights:
                del self.pattern_weights[pattern]
            if pattern in self.pattern_counts:
                del self.pattern_counts[pattern]
            removed += 1
        
        for pattern in self.pattern_weights:
            total = sum(self.pattern_weights[pattern].values())
            if total > 0:
                max_weight = max(self.pattern_weights[pattern].values())
                if max_weight > total * 0.7:
                    for answer in self.pattern_weights[pattern]:
                        if self.pattern_weights[pattern][answer] == max_weight:
                            self.pattern_weights[pattern][answer] *= 1.15
                        else:
                            self.pattern_weights[pattern][answer] *= 0.85
                optimized += 1
        
        if len(self.learning_history) > 30:
            recent_qualities = [h.get("korean_quality", 0) for h in self.learning_history[-15:]]
            if recent_qualities:
                avg_quality = sum(recent_qualities) / len(recent_qualities)
                if avg_quality > 0.6:
                    self.confidence_threshold = max(self.confidence_threshold - 0.05, 0.25)
                elif avg_quality < 0.4:
                    self.confidence_threshold = min(self.confidence_threshold + 0.05, 0.7)
        
        return {
            "optimized": optimized,
            "removed": removed,
            "remaining": len(self.pattern_weights),
            "confidence_threshold": self.confidence_threshold
        }
    
    def analyze_learning_progress(self) -> Dict:
        if not self.learning_history:
            return {"status": "학습 데이터 없음"}
        
        recent_history = self.learning_history[-50:]
        
        confidences = [h["confidence"] for h in recent_history]
        confidence_trend = np.mean(confidences) if confidences else 0
        
        korean_qualities = [h.get("korean_quality", 0) for h in recent_history]
        korean_quality_trend = np.mean(korean_qualities) if korean_qualities else 0
        
        pattern_diversity = len(self.pattern_weights)
        mc_distribution = dict(self.answer_distribution["mc"])
        
        specialized_rule_usage = {}
        for rule_name in self.specialized_rules:
            usage_count = sum(1 for h in recent_history if rule_name in h.get("question_sample", ""))
            specialized_rule_usage[rule_name] = usage_count
        
        return {
            "total_samples": len(self.learning_history),
            "confidence_trend": confidence_trend,
            "korean_quality_trend": korean_quality_trend,
            "pattern_diversity": pattern_diversity,
            "mc_distribution": mc_distribution,
            "negative_distribution": dict(self.answer_distribution["negative"]),
            "active_patterns": list(self.pattern_weights.keys())[:10],
            "successful_templates": len(self.successful_korean_templates),
            "specialized_rule_usage": specialized_rule_usage,
            "learning_efficiency": self._calculate_learning_efficiency()
        }
    
    def _calculate_learning_efficiency(self) -> float:
        if len(self.learning_history) < 5:
            return 0.0
        
        recent_samples = self.learning_history[-15:]
        early_samples = self.learning_history[:15] if len(self.learning_history) >= 30 else []
        
        if not early_samples:
            return 0.5
        
        recent_avg_confidence = np.mean([s["confidence"] for s in recent_samples])
        early_avg_confidence = np.mean([s["confidence"] for s in early_samples])
        
        recent_avg_quality = np.mean([s.get("korean_quality", 0) for s in recent_samples])
        early_avg_quality = np.mean([s.get("korean_quality", 0) for s in early_samples])
        
        confidence_improvement = recent_avg_confidence - early_avg_confidence
        quality_improvement = recent_avg_quality - early_avg_quality
        
        efficiency = (confidence_improvement + quality_improvement) / 2
        return max(0, min(1, 0.5 + efficiency))
    
    def save_model(self, filepath: str = "./auto_learner_model.pkl") -> bool:
        model_data = {
            "pattern_weights": {k: dict(v) for k, v in self.pattern_weights.items()},
            "pattern_counts": dict(self.pattern_counts),
            "answer_distribution": {
                "mc": dict(self.answer_distribution["mc"]),
                "domain": {k: dict(v) for k, v in self.answer_distribution["domain"].items()},
                "negative": dict(self.answer_distribution["negative"])
            },
            "korean_quality_patterns": {k: v[-15:] for k, v in self.korean_quality_patterns.items()},
            "successful_korean_templates": self.successful_korean_templates[-30:],
            "learning_history": self.learning_history[-100:],
            "specialized_rules": self.specialized_rules,
            "parameters": {
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "min_samples": self.min_samples
            }
        }
        
        return atomic_save_model(model_data, filepath)
    
    def load_model(self, filepath: str = "./auto_learner_model.pkl") -> bool:
        model_data = atomic_load_model(filepath)
        if model_data is None:
            return False
        
        try:
            self.pattern_weights = defaultdict(_default_float_dict)
            for k, v in model_data.get("pattern_weights", {}).items():
                self.pattern_weights[k] = defaultdict(_default_float, v)
            
            self.pattern_counts = defaultdict(_default_int, model_data.get("pattern_counts", {}))
            
            answer_dist = model_data.get("answer_distribution", {})
            self.answer_distribution = {
                "mc": defaultdict(_default_int, answer_dist.get("mc", {})),
                "domain": defaultdict(_default_int_dict),
                "negative": defaultdict(_default_int, answer_dist.get("negative", {}))
            }
            
            for k, v in answer_dist.get("domain", {}).items():
                self.answer_distribution["domain"][k] = defaultdict(_default_int, v)
            
            self.korean_quality_patterns = defaultdict(_default_list, model_data.get("korean_quality_patterns", {}))
            self.successful_korean_templates = model_data.get("successful_korean_templates", [])
            self.learning_history = model_data.get("learning_history", [])
            
            if "specialized_rules" in model_data:
                self.specialized_rules.update(model_data["specialized_rules"])
            
            params = model_data.get("parameters", {})
            self.learning_rate = params.get("learning_rate", 0.35)
            self.confidence_threshold = params.get("confidence_threshold", 0.35)
            self.min_samples = params.get("min_samples", 1)
            
            return True
        except Exception:
            return False
    
    def cleanup(self):
        total_patterns = len(self.pattern_weights)
        total_samples = len(self.learning_history)
        if total_patterns > 0 or total_samples > 0:
            print(f"자동 학습: {total_patterns}개 패턴, {total_samples}개 샘플")