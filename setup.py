#!/usr/bin/env python3
"""
Debug Sign Language to Voice Converter
Enhanced version with robust TTS handling
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pickle
import os
import time
import threading
from sklearn.ensemble import RandomForestClassifier
from queue import Queue

class DebugSignToVoice:
    def __init__(self):
        print("🔧 Initializing Debug Sign Language Converter...")
        
        # Initialize MediaPipe with more lenient settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # TTS setup with threading
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.speech_active = False
        self.setup_tts()
        
        # Data storage
        self.signs_data = {}
        self.model = None
        
        # Detection settings
        self.last_spoken_time = 0
        self.last_spoken_word = None
        self.speech_cooldown = 1.5
        
        # Debug settings
        self.debug_mode = True
        
        # Load data
        self.load_data()
        print("✅ Initialization complete!")
    
    def setup_tts(self):
        """Setup TTS with multiple attempts and testing"""
        print("\n🔊 Setting up Text-to-Speech...")
        
        for attempt in range(3):
            try:
                print(f"  Attempt {attempt + 1}/3...")
                self.tts_engine = pyttsx3.init()
                
                # Configure TTS
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)
                
                # Try to get voices
                voices = self.tts_engine.getProperty('voices')
                if voices and len(voices) > 0:
                    self.tts_engine.setProperty('voice', voices[0].id)
                    print(f"  ✅ Voice set: {voices[0].name}")
                
                # CRITICAL: Test TTS immediately in a controlled way
                print("  🧪 Testing TTS...")
                test_thread = threading.Thread(target=self._test_tts_thread)
                test_thread.start()
                test_thread.join(timeout=3)
                
                if test_thread.is_alive():
                    print("  ⚠ TTS test timed out")
                    self.tts_engine = None
                    continue
                
                print("  ✅ TTS is working!")
                return
                
            except Exception as e:
                print(f"  ❌ TTS setup failed: {e}")
                self.tts_engine = None
                time.sleep(0.5)
        
        print("  ⚠ TTS unavailable - will use text output only")
    
    def _test_tts_thread(self):
        """Test TTS in separate thread"""
        try:
            self.tts_engine.say("Ready")
            self.tts_engine.runAndWait()
        except:
            pass
    
    def debug_print(self, message):
        """Print debug messages"""
        if self.debug_mode:
            print(f"🔍 DEBUG: {message}")
    
    def extract_features(self, landmarks):
        """Extract features with debugging"""
        if not landmarks:
            self.debug_print("No landmarks provided")
            return None
        
        try:
            # Basic coordinates
            features = []
            for i, lm in enumerate(landmarks.landmark):
                features.extend([lm.x, lm.y])
            
            # Convert to numpy for distance calculations
            points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            
            # Key landmarks
            thumb_tip = points[4]
            index_tip = points[8]  
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]
            wrist = points[0]
            
            # Calculate distances
            distances = [
                np.linalg.norm(thumb_tip - index_tip),
                np.linalg.norm(thumb_tip - middle_tip),
                np.linalg.norm(index_tip - middle_tip),
                np.linalg.norm(middle_tip - ring_tip),
                np.linalg.norm(ring_tip - pinky_tip),
                np.linalg.norm(thumb_tip - wrist),
                np.linalg.norm(index_tip - wrist)
            ]
            
            features.extend(distances)
            result = np.array(features)
            
            return result
            
        except Exception as e:
            self.debug_print(f"Feature extraction error: {e}")
            return None
    
    def speak(self, word, bypass_cooldown=False):
        """Speak word using direct threading approach"""
        current_time = time.time()
        
        # Check cooldown
        if not bypass_cooldown:
            if current_time - self.last_spoken_time < self.speech_cooldown:
                self.debug_print(f"Speech cooldown active for '{word}'")
                return False
            
            # If same word, require longer cooldown
            if word == self.last_spoken_word:
                if current_time - self.last_spoken_time < self.speech_cooldown * 2:
                    self.debug_print(f"Repeat word cooldown for '{word}'")
                    return False
        
        # Update tracking immediately
        self.last_spoken_time = current_time
        self.last_spoken_word = word
        
        print(f"🗣 SPEAKING: {word.upper()}")
        
        # Speak in background thread
        if self.tts_engine is not None:
            speech_thread = threading.Thread(target=self._speak_worker, args=(word,))
            speech_thread.daemon = True
            speech_thread.start()
            return True
        else:
            print(f"   >>> {word.upper()} <<<")
            return True
    
    def _speak_worker(self, word):
        """Worker thread for speaking"""
        try:
            with self.tts_lock:
                # Create fresh engine instance for this thread
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(word)
                engine.runAndWait()
                engine.stop()
                del engine
        except Exception as e:
            print(f"  ⚠ Speech error: {e}")
    
    def capture_sign_simple(self, word, num_samples=5):
        """Simplified capture with more debugging"""
        print(f"\n📸 CAPTURING: {word}")
        print("Position your hand clearly and press SPACE to capture")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return False
        
        try:
            captured_features = []
            count = 0
            
            print("Camera opened. Show your hand gesture...")
            
            while count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                # Large, clear text
                cv2.putText(frame, f"WORD: {word.upper()}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"SAMPLES: {count}/{num_samples}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                hand_detected = False
                if results.multi_hand_landmarks:
                    hand_detected = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                    
                    cv2.putText(frame, "HAND DETECTED - PRESS SPACE!", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "SHOW YOUR HAND CLEARLY", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.putText(frame, "SPACE=Capture, Q=Quit", (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Capture Sign - Debug Mode', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space pressed
                    if hand_detected and results.multi_hand_landmarks:
                        features = self.extract_features(results.multi_hand_landmarks[0])
                        if features is not None:
                            captured_features.append(features)
                            count += 1
                            print(f"✅ CAPTURED sample {count}/{num_samples}")
                            
                            # Visual feedback
                            feedback_frame = frame.copy()
                            cv2.putText(feedback_frame, "CAPTURED!", (200, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                            cv2.imshow('Capture Sign - Debug Mode', feedback_frame)
                            cv2.waitKey(800)
                        else:
                            print("❌ Failed to extract features")
                    else:
                        print("❌ No hand detected - position hand clearly")
                
                elif key == ord('q'):
                    print("Capture cancelled by user")
                    break
            
            if captured_features:
                self.signs_data[word] = captured_features
                print(f"✅ Successfully captured {len(captured_features)} samples for '{word}'")
                return True
            else:
                print("❌ No samples captured!")
                return False
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def train_model_simple(self):
        """Simplified training with debugging"""
        if len(self.signs_data) < 1:
            print("❌ No training data available!")
            return False
            
        if len(self.signs_data) < 2:
            print("⚠ Only 1 word available. Training anyway for testing...")
        
        print("🤖 TRAINING MODEL...")
        
        # Prepare data
        X, y = [], []
        for word, features_list in self.signs_data.items():
            print(f"  📊 {word}: {len(features_list)} samples")
            for features in features_list:
                X.append(features)
                y.append(word)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📈 Training data: {X.shape[0]} samples, {len(np.unique(y))} classes")
        self.debug_print(f"Feature matrix shape: {X.shape}")
        self.debug_print(f"Labels: {np.unique(y)}")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X, y)
        print("✅ MODEL TRAINED!")
        
        # Test model on training data
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"📊 Training accuracy: {accuracy:.2f}")
        
        self.save_data()
        return True
    
    def predict_sign_simple(self, features):
        """Simplified prediction with debugging"""
        if self.model is None:
            self.debug_print("No model available for prediction")
            return None, 0
            
        if features is None:
            self.debug_print("No features provided for prediction")
            return None, 0
        
        try:
            # Reshape features
            features_2d = features.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(features_2d)[0]
            probabilities = self.model.predict_proba(features_2d)[0]
            confidence = np.max(probabilities)
            
            self.debug_print(f"Prediction: {prediction}, confidence: {confidence:.3f}")
            
            # Lower threshold for detection
            if confidence > 0.25:
                return prediction, confidence
            else:
                self.debug_print(f"Confidence too low: {confidence:.3f}")
                
        except Exception as e:
            self.debug_print(f"Prediction error: {e}")
        
        return None, 0
    
    def run_detection_simple(self):
        """Simplified detection with voice output"""
        if self.model is None:
            print("❌ No trained model! Train first using option 3.")
            return
        
        print("🎥 STARTING DETECTION (Debug Mode)")
        print("Show your trained gestures clearly")
        print("Press 'q' to quit, 'd' to toggle debug, 't' to test speech")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera error!")
            return
        
        try:
            frame_count = 0
            detection_count = 0
            speech_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(frame_rgb)
                
                predicted_word = None
                confidence = 0
                
                if results.multi_hand_landmarks:
                    detection_count += 1
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks with thick lines
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)
                        )
                        
                        # Extract features and predict
                        features = self.extract_features(hand_landmarks)
                        if features is not None:
                            predicted_word, confidence = self.predict_sign_simple(features)
                
                # Display information
                if predicted_word:
                    # Large text for prediction
                    cv2.putText(frame, f"DETECTED: {predicted_word.upper()}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    # Try to speak if confidence is good
                    if confidence > 0.25:
                        current_time = time.time()
                        time_since_last = current_time - self.last_spoken_time
                        
                        if time_since_last > self.speech_cooldown:
                            spoke = self.speak(predicted_word)
                            if spoke:
                                speech_count += 1
                                cv2.putText(frame, "SPEAKING NOW!", (10, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        else:
                            remaining = self.speech_cooldown - time_since_last
                            cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
                
                # Debug info
                cv2.putText(frame, f"Frames: {frame_count}", (10, frame.shape[0] - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"Hand detections: {detection_count}", (10, frame.shape[0] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"Speech attempts: {speech_count}", (10, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"TTS: {'Active' if self.tts_engine else 'Disabled'}", 
                            (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Known words
                words = ", ".join(self.signs_data.keys()) if self.signs_data else "None"
                cv2.putText(frame, f"Known: {words}", (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.putText(frame, "Q=Quit, D=Debug, T=Test Speech", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Detection - Debug Mode', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('t'):
                    print("Testing speech...")
                    self.speak("test", bypass_cooldown=True)
            
            print(f"\nSession stats:")
            print(f"  Total frames: {frame_count}")
            print(f"  Hand detections: {detection_count}")
            print(f"  Speech attempts: {speech_count}")
            print(f"  Detection rate: {detection_count/frame_count*100:.1f}%" if frame_count > 0 else "  Detection rate: 0%")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def save_data(self):
        """Save data with error handling"""
        try:
            with open('debug_sign_data.pkl', 'wb') as f:
                pickle.dump({
                    'signs_data': self.signs_data,
                    'model': self.model
                }, f)
            print("💾 Data saved to debug_sign_data.pkl")
        except Exception as e:
            print(f"Save error: {e}")
    
    def load_data(self):
        """Load data with error handling"""
        try:
            with open('debug_sign_data.pkl', 'rb') as f:
                data = pickle.load(f)
                self.signs_data = data.get('signs_data', {})
                self.model = data.get('model', None)
            if self.signs_data:
                print(f"✅ Loaded data for: {list(self.signs_data.keys())}")
            else:
                print("ℹ No existing data found")
        except FileNotFoundError:
            print("ℹ Starting with fresh data")
        except Exception as e:
            print(f"Load error: {e}")
    
    def test_camera(self):
        """Test camera and hand detection"""
        print("🎥 TESTING CAMERA AND HAND DETECTION")
        print("Show your hand to test detection. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        try:
            detection_count = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    detection_count += 1
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                    
                    cv2.putText(frame, "HAND DETECTED!", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Detection rate: {detection_count/frame_count*100:.1f}%", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Camera Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            print(f"\nCamera test results:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Hand detections: {detection_count}")
            print(f"  Detection rate: {detection_count/frame_count*100:.1f}%" if frame_count > 0 else "  No frames processed")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def test_tts_manual(self):
        """Manual TTS test"""
        print("\n🔊 Testing TTS Manually...")
        
        if self.tts_engine is None:
            print("❌ TTS engine is not initialized!")
            print("Trying to reinitialize...")
            self.setup_tts()
            
        if self.tts_engine:
            print("Attempting to speak 'Hello World'...")
            self.speak("Hello World", bypass_cooldown=True)
            time.sleep(2)
            print("Speech command sent. Did you hear it?")
        else:
            print("❌ TTS still not available")
    
    def menu(self):
        """Debug menu"""
        while True:
            print("\n" + "="*50)
            print("🔧 DEBUG Sign Language Converter")
            print("="*50)
            
            if self.signs_data:
                print(f"📚 Words learned: {list(self.signs_data.keys())}")
                for word, samples in self.signs_data.items():
                    print(f"    - {word}: {len(samples)} samples")
            else:
                print("📚 No words learned yet")
            
            print(f"🤖 Model trained: {'Yes' if self.model else 'No'}")
            print(f"🗣 TTS available: {'Yes' if self.tts_engine else 'No'}")
            print(f"🔍 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            
            print("""
1. 🎥 Test camera & hand detection
2. 📸 Capture sign (simple mode)  
3. 🤖 Train model (simple mode)
4. 🔍 Run detection (debug mode)
5. 💾 Save/Load status
6. 🔊 Test TTS manually
7. ❌ Exit
""")
            
            choice = input("Select (1-7): ").strip()
            
            if choice == '1':
                self.test_camera()
                
            elif choice == '2':
                word = input("Word to teach: ").strip().lower()
                if word:
                    self.capture_sign_simple(word)
                    
            elif choice == '3':
                self.train_model_simple()
                
            elif choice == '4':
                self.run_detection_simple()
                
            elif choice == '5':
                print(f"Data file exists: {os.path.exists('debug_sign_data.pkl')}")
                print(f"Words in memory: {len(self.signs_data)}")
                print(f"Model in memory: {self.model is not None}")
                
            elif choice == '6':
                self.test_tts_manual()
                
            elif choice == '7':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")

if __name__ == "__main__":
    converter = DebugSignToVoice()
    converter.menu()