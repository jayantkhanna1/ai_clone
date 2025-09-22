// Global variables
let isRecording = false;
let recognition = null;
let speechSynthesis = window.speechSynthesis;
let currentUtterance = null;
let isPaused = false;
let currentTheme = 'light';
let BACKEND_URL = "https://aiclone.jayantkhanna.in";

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeSpeechRecognition();
    checkBrowserCompatibility();
    requestMicrophonePermissionOnFirstClick();
    
    // Load voices for speech synthesis
    if (speechSynthesis) {
        speechSynthesis.onvoiceschanged = function() {
            console.log('Voices loaded:', speechSynthesis.getVoices().length);
        };
    }
});

// Enhanced browser compatibility check
function checkBrowserCompatibility() {
    const userAgent = navigator.userAgent.toLowerCase();
    const isBrave = userAgent.includes('brave') || (typeof navigator.brave !== 'undefined' && navigator.brave.isBrave);
    const isFirefox = userAgent.includes('firefox');
    const isSafari = userAgent.includes('safari') && !userAgent.includes('chrome');
    const isEdge = userAgent.includes('edge');
    
    const hasSpeechRecognition = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    const hasSpeechSynthesis = 'speechSynthesis' in window;
    
    console.log('Browser Detection:', {
        userAgent,
        isBrave,
        isFirefox,
        isSafari,
        isEdge,
        hasSpeechRecognition,
        hasSpeechSynthesis
    });

    if (isBrave || isFirefox || isSafari || !hasSpeechRecognition || !hasSpeechSynthesis) {
        const warning = document.getElementById('browserWarning');
        warning.classList.add('show');
        setTimeout(() => {
            warning.classList.remove('show');
        }, 10000);
    }

    return {
        supported: hasSpeechRecognition && hasSpeechSynthesis,
        browser: isBrave ? 'brave' : isFirefox ? 'firefox' : isSafari ? 'safari' : isEdge ? 'edge' : 'chrome'
    };
}

// Initialize speech recognition
function initializeSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = function() {
        console.log('Speech recognition started');
        setStatus('Listening...');
        document.getElementById('orbContainer').classList.add('listening-state');
    };

    recognition.onresult = function(event) {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        const currentTranscript = finalTranscript || interimTranscript;
        document.getElementById('questionInput').value = currentTranscript;

        if (finalTranscript) {
            console.log('Final transcript:', finalTranscript);
            stopRecording();
            if (finalTranscript.trim()) {
                askQuestion();
            }
        }
    };

    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        showError('Voice recognition failed: ' + event.error);
        resetVoiceButton();
        
        switch(event.error) {
            case 'not-allowed':
                alert('Microphone access denied. Please allow microphone access and try again.');
                break;
            case 'no-speech':
                alert('No speech detected. Please try again.');
                break;
            case 'audio-capture':
                alert('Audio capture failed. Please check your microphone.');
                break;
            case 'network':
                alert('Network error. Please check your internet connection.');
                break;
        }
    };

    recognition.onend = function() {
        console.log('Speech recognition ended');
        resetVoiceButton();
    };
}

// Theme functions
function initializeTheme() {
    try {
        const savedTheme = localStorage.getItem('theme') || 'light';
        setTheme(savedTheme);
    } catch (error) {
        console.log('LocalStorage not available, using default theme');
        setTheme('light');
    }
}

function toggleTheme() {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    
    try {
        localStorage.setItem('theme', newTheme);
    } catch (error) {
        console.log('Could not save theme preference');
    }
}

function setTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);

    const themeIcon = document.getElementById('themeIcon');
    const themeText = document.getElementById('themeText');

    if (theme === 'dark') {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light';
    } else {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark';
    }
}

// Input handling
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}

// Voice recording functions
function toggleVoiceRecording() {
    if (!recognition) {
        alert('Speech recognition is not available in your browser. Please try using Google Chrome.');
        return;
    }

    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (!recognition) return;

    try {
        isRecording = true;
        const voiceBtn = document.getElementById('voiceBtn');
        const voiceBtnText = document.getElementById('voiceBtnText');
        const orbContainer = document.getElementById('orbContainer');

        voiceBtn.classList.add('recording');
        voiceBtnText.textContent = '⏹️ Recording';
        orbContainer.classList.add('listening-state');
        
        document.getElementById('questionInput').value = '';
        setStatus('Listening...');
        setOrbText('Listening');
        
        recognition.start();
    } catch (error) {
        console.error('Failed to start recording:', error);
        showError('Failed to start voice recording');
        resetVoiceButton();
    }
}

function stopRecording() {
    if (!recognition) return;

    isRecording = false;
    recognition.stop();
    resetVoiceButton();
    setStatus('Processing...');
    setOrbText('Processing');
}

function resetVoiceButton() {
    isRecording = false;
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceBtnText = document.getElementById('voiceBtnText');
    const orbContainer = document.getElementById('orbContainer');

    voiceBtn.classList.remove('recording');
    voiceBtnText.textContent = '🎤 Voice';
    orbContainer.classList.remove('listening-state');
}

// UI helper functions
function setStatus(text) {
    const statusText = document.getElementById('statusText');
    statusText.textContent = text;
    statusText.classList.add('show');
}

function setOrbText(text) {
    const orbText = document.getElementById('orbText');
    orbText.textContent = text;
}

function showAnswerDisplay(text) {
    const answerDisplay = document.getElementById('answerDisplay');
    const answerDisplayText = document.getElementById('answerDisplayText');
    answerDisplayText.textContent = text;
    answerDisplay.classList.add('show');
}

function hideAnswerDisplay() {
    const answerDisplay = document.getElementById('answerDisplay');
    answerDisplay.classList.remove('show');
}

// Main question handling with backend integration
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();

    if (!question) {
        showError('Please enter a question.');
        return;
    }

    showLoading();

    try {
        const response = await fetch(BACKEND_URL + '/answerQuestion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const answer = data.answer || data.response || data.text || JSON.stringify(data);
        
        animateAnswer(answer);
        questionInput.value = ''; // Clear input after successful submission

    } catch (error) {
        console.error('Backend connection error:', error);
        
        // Fallback to local responses if backend is not available
        console.log('Using fallback response generation');
        const fallbackAnswer = generateFallbackResponse(question);
        animateAnswer(fallbackAnswer);
        questionInput.value = '';
    }
}

// Fallback response generator (used when backend is not available)
function generateFallbackResponse(question) {
    const responses = [
        "That's a great question! Based on current knowledge, I'd say the answer involves multiple factors that we should consider carefully.",
        "Interesting! This topic has been studied extensively, and the research suggests several important points to consider.",
        "Hello! I'm Jayant Khanna, a passionate software engineer focused on building scalable solutions and exploring innovative technologies to solve real-world problems.",
        "Thanks for asking! The answer depends on several variables, but generally speaking, the best approach would be to analyze the situation systematically.",
        "That's an excellent point to explore! From my understanding, this involves both technical and practical considerations that we should examine.",
        "Great question! This is something I've been thinking about lately, and I believe the key lies in understanding the fundamental principles involved.",
        "I appreciate you asking! Based on current trends and data, I would recommend considering multiple perspectives on this topic."
    ];
    
    const lowerQuestion = question.toLowerCase();
    
    if (lowerQuestion.includes('who are you') || lowerQuestion.includes('about you') || lowerQuestion.includes('yourself')) {
        return "Hello! I'm Jayant Khanna, a passionate software engineer focused on building scalable solutions and exploring innovative technologies to solve real-world problems.";
    }
    
    if (lowerQuestion.includes('how are you') || lowerQuestion.includes('how do you feel')) {
        return "I'm doing great, thank you for asking! I'm excited to help you with any questions you might have.";
    }
    
    if (lowerQuestion.includes('weather') || lowerQuestion.includes('temperature')) {
        return "I don't have access to real-time weather data, but I'd recommend checking your local weather app or website for the most accurate information.";
    }
    
    if (lowerQuestion.includes('time') || lowerQuestion.includes('date')) {
        return "I don't have access to real-time data, but you can check the current time and date on your device or computer.";
    }
    
    return responses[Math.floor(Math.random() * responses.length)];
}

function showLoading() {
    const orbContainer = document.getElementById('orbContainer');
    const controls = document.getElementById('controls');

    controls.classList.remove('show');
    orbContainer.className = 'orb-container loading-state';
    stopCurrentSpeech();
    hideAnswerDisplay();
    setStatus('Processing...');
    setOrbText('Thinking');
}

function showError(message) {
    const orbContainer = document.getElementById('orbContainer');
    const controls = document.getElementById('controls');

    controls.classList.remove('show');
    orbContainer.className = 'orb-container error-state';
    stopCurrentSpeech();
    hideAnswerDisplay();
    setStatus('Error');
    setOrbText('Error');

    setTimeout(() => {
        orbContainer.className = 'orb-container';
        setStatus('Ready');
        setOrbText('Ready');
    }, 2000);
}

function animateAnswer(answer) {
    const orbContainer = document.getElementById('orbContainer');
    const controls = document.getElementById('controls');
    const mainOrb = document.getElementById('mainOrb');

    stopCurrentSpeech();
    orbContainer.className = 'orb-container';
    
    // Always show the answer display first, regardless of speech synthesis
    showAnswerDisplay(answer);
    controls.classList.add('show');
    
    // Try to speak the text, but don't let speech synthesis failure affect the display
    const speechSuccessful = speakText(answer);
    
    if (speechSuccessful) {
        // Speech synthesis is working
        mainOrb.classList.add('orb-speaking');
        setStatus('Speaking...');
        setOrbText('Speaking');
    } else {
        // Speech synthesis failed, but still show the answer
        console.warn('Speech synthesis failed, but answer is still displayed');
        setStatus('Answer Ready');
        setOrbText('Answer Ready');
        
        // // Auto-hide after a delay since there's no speech to wait for
        // setTimeout(() => {
        //     controls.classList.remove('show');
        //     setStatus('Complete');
        //     setOrbText('Ready');
        //     setTimeout(() => {
        //         hideAnswerDisplay();
        //         setStatus('Ready');
        //     }, 3000);
        // }, 5000); // Show answer for 5 seconds
    }
}

// Enhanced speech synthesis function with better error handling
function speakText(text) {
    try {
        if (!text) {
            console.warn('No text provided for speech synthesis');
            return false;
        }

        if (!speechSynthesis) {
            console.warn('Speech synthesis not available');
            return false;
        }

        // Check if speech synthesis is supported and working
        if (speechSynthesis.getVoices().length === 0) {
            console.warn('No voices available for speech synthesis');
            // Try to load voices and speak after a delay
            setTimeout(() => {
                if (speechSynthesis.getVoices().length > 0) {
                    attemptSpeechSynthesis(text);
                } else {
                    console.warn('Voices still not available after delay');
                }
            }, 100);
            return false;
        }

        return attemptSpeechSynthesis(text);

    } catch (error) {
        console.error('Speech synthesis setup failed:', error);
        return false;
    }
}

// Separate function to attempt speech synthesis
function attemptSpeechSynthesis(text) {
    try {
        currentUtterance = new SpeechSynthesisUtterance(text);
        
        // Enhanced voice selection
        const voices = speechSynthesis.getVoices();
        const preferredVoices = voices.filter(voice => 
            voice.lang.startsWith('en') && 
            (voice.name.includes('Google') || voice.name.includes('Microsoft') || voice.default)
        );
        
        if (preferredVoices.length > 0) {
            currentUtterance.voice = preferredVoices[0];
        } else if (voices.length > 0) {
            currentUtterance.voice = voices.find(voice => voice.default) || voices[0];
        }

        currentUtterance.rate = 0.8;
        currentUtterance.pitch = 1;
        currentUtterance.volume = 0.8;

        currentUtterance.onstart = function() {
            console.log('Speech started');
        };

        currentUtterance.onend = function() {
            console.log('Speech ended normally');
            handleSpeechEnd();
        };

        currentUtterance.onerror = function(event) {
            console.error('Speech synthesis error during playback:', event.error);
            // Don't show error to user since answer is already displayed
            // Just clean up and treat as if speech ended
            handleSpeechEnd();
        };

        // Attempt to speak
        speechSynthesis.speak(currentUtterance);
        
        // Set a timeout as a fallback in case speech events don't fire
        setTimeout(() => {
            if (currentUtterance && speechSynthesis.speaking) {
                // Speech is still going, let it continue
                return;
            } else if (currentUtterance) {
                // Speech might have failed silently
                console.warn('Speech synthesis timeout - cleaning up');
                handleSpeechEnd();
            }
        }, 30000); // 30 second timeout

        return true;

    } catch (error) {
        console.error('Speech synthesis playback failed:', error);
        return false;
    }
}

// Handle speech end (both successful and failed cases)
function handleSpeechEnd() {
    currentUtterance = null;
    const controls = document.getElementById('controls');
    const mainOrb = document.getElementById('mainOrb');
    
    controls.classList.remove('show');
    mainOrb.classList.remove('orb-speaking');
    setStatus('Complete');
    setOrbText('Ready');
    
    setTimeout(() => {
        hideAnswerDisplay();
        setStatus('Ready');
    }, 3000);
}

// Speech control functions
function pauseResume() {
    const pauseBtn = document.getElementById('pauseBtn');
    const mainOrb = document.getElementById('mainOrb');

    if (!speechSynthesis) return;

    if (speechSynthesis.paused) {
        speechSynthesis.resume();
        pauseBtn.textContent = 'Pause';
        setStatus('Speaking...');
        setOrbText('Speaking');
        mainOrb.classList.add('orb-speaking');
        isPaused = false;
    } else {
        speechSynthesis.pause();
        pauseBtn.textContent = 'Resume';
        setStatus('Paused');
        setOrbText('Paused');
        mainOrb.classList.remove('orb-speaking');
        isPaused = true;
    }
}

function stopSpeech() {
    const mainOrb = document.getElementById('mainOrb');
    const controls = document.getElementById('controls');

    stopCurrentSpeech();
    controls.classList.remove('show');
    mainOrb.classList.remove('orb-speaking');
    setStatus('Stopped');
    setOrbText('Ready');
    hideAnswerDisplay();

    setTimeout(() => {
        setStatus('Ready');
    }, 1000);
}

function stopCurrentSpeech() {
    if (speechSynthesis) {
        speechSynthesis.cancel();
    }
    if (currentUtterance) {
        currentUtterance = null;
    }
    isPaused = false;
    const pauseBtn = document.getElementById('pauseBtn');
    if (pauseBtn) {
        pauseBtn.textContent = 'Pause';
    }
}

// Microphone permission handling
function requestMicrophonePermissionOnFirstClick() {
    document.addEventListener('click', function() {
        requestMicrophonePermission();
    }, { once: true });
}

function requestMicrophonePermission() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                console.log('Microphone permission granted');
                stream.getTracks().forEach(track => track.stop());
            })
            .catch(function(error) {
                console.error('Microphone permission denied:', error);
                alert('Microphone access is required for voice input. Please allow microphone access and refresh the page.');
            });
    }
}

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        if (speechSynthesis && speechSynthesis.speaking && !speechSynthesis.paused) {
            speechSynthesis.pause();
        }
    } else {
        if (speechSynthesis && speechSynthesis.paused) {
            speechSynthesis.resume();
        }
    }
});

// Cleanup before page unload
window.addEventListener('beforeunload', function() {
    if (speechSynthesis) {
        speechSynthesis.cancel();
    }
    if (recognition && isRecording) {
        recognition.stop();
    }
});

// Debug information
console.log('Voice Assistant initialized');
console.log('Browser support:', {
    SpeechRecognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
    SpeechSynthesis: 'speechSynthesis' in window,
    getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
});
console.log('Backend URL:', BACKEND_URL);