"""
Online ASR processor for streaming transcription
"""

import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None
        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer
        new = [(a+offset,b+offset,t) for a,b,t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # Search for 1-5 consecutive identical words
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer

class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """Initialize the online ASR processor.
        
        Args:
            asr: WhisperASR object
            buffer_trimming: Tuple of (option, seconds), where option is "segment" and seconds is buffer length threshold
            logfile: Where to store the log
        """
        self.asr = asr
        self.logfile = logfile
        self.init()
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """Initialize or reset the processor state"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        """Insert a new chunk of audio for processing"""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Get the prompt and context for transcription"""
        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        """Process the current audio buffer.
        Returns: (beg_timestamp, end_timestamp, "text") or (None, None, "")
        """
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        tsw = self.asr.ts_words(res)
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:
            self.chunk_completed_segment(res)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_segment(self, res):
        if self.commited == []: return

        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2]+self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time):
        """Trim the hypothesis and audio buffer at given time"""
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def finish(self):
        """Flush incomplete text when processing ends"""
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f

    def to_flush(self, sents, sep=None, offset=0):
        """Format timestamped words/sentences for output"""
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t) 