import unittest

from app import clean_text, predict_emotion

class TestEmoInSync(unittest.TestCase):

    def test_clean_text(self):
        self.assertEqual(clean_text("This is a sample text!"), "sampl text")
        self.assertEqual(clean_text("Example with stopwords."), "exampl stopword")

    def test_predict_emotion(self):
        text1 = "I am feeling happy today."
        emotion1, label1 = predict_emotion(text1)
        self.assertEqual(emotion1, "joy")
        self.assertEqual(label1, 2)

        text2 = "I am feeling sad."
        emotion2, label2 = predict_emotion(text2)
        self.assertEqual(emotion2, "sadness")
        self.assertEqual(label2, 4)

if __name__ == '__main__':
    unittest.main()