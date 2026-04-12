# TinyStories Generation Comparison

Side-by-side generations from three models trained on TinyStories (A100).

| Model | Params | Field | Step | d_model |
|-------|--------|-------|------|---------|
| baseline | 29.9M | No | 8000 | 384 |
| social | 30.8M | Yes (additive) | 8000 | 384 |
| baseline_1_8x | 52.8M | No | 10000 | 576 |

---

## Prompt: "The dog found a shiny key"

### Baseline (30M) (step 8000)

The dog found a shiny key. It asked, "Do you have any gold key?"

The dog said, "I found it in a pond!"

Ceddy was very excited. He went to the pond and started to collect gold coins. He collected many spark coins and put them in his pocket. Then he went back to the pond and showed his mom the coins. He wanted to show her his treasure.

Ceddy knew that the coin was very special to him. He took it home and showed it to his mom. She said, "That is a very special coin, but it is also very special. You should always be careful with it."

Ceddy nodded and said, "I will. Thank you, mom."

He decided to keep the coin and kept it safe. He was very happy. He thought of more gold coins and how much he loved them. He showed his mom his treasure and said, "Look, mom! I collected coins and I am very proud of my coin. I will show it to everyone."

### Social Field (31M) (step 8000)

The dog found a shiny key in the mailbox. The owner of the dog was very happy and thanked the girl.

The next day, the owner went back to the dog's house. The car was very fast and did not see the big cat. The dog saw the key and thought it was a toy. The dog jumped on the car and started to chase the cat. The car ran and ran, but the key was too fast. The dog caught the key and gave it back to the owner of the house. The owner was very happy to have the key back, and he thanked the boy. The boy and the dog became good friends, and they played together every day.

### Baseline 1.8x (53M) (step 10000)

The dog found a shiny key. It was so excited! 

The dog put the key in his pocket and ran off to find more adventures. He was so excited to have found the big, shiny thing again!

---

## Prompt: "Sara gave her teddy bear to Tom"

### Baseline (30M) (step 8000)

Sara gave her teddy bear to Tom. Tom was happy. He hugged Mia back. He said, "Thank you, thank you, Mia. I love you."

The end.

### Social Field (31M) (step 8000)

Sara gave her teddy bear to Tom. Tom hugged her back and said, "Thank you, Sue. You are a good friend."

Sara smiled and said, "You're welcome, Tom. You're a good friend too, Tom. You helped me make your teddy bear dance."

They played with their toys and sang songs. They were happy. They were friends.

### Baseline 1.8x (53M) (step 10000)

Sara gave her teddy bear to Tom. Tom was very happy and hugged Lily. He said, "Thank you, Lily. You are very kind. Can I see your teddy bear?"

Lily nodded and gave Tom her teddy bear. Tom looked at his teddy bear and felt jealous. He wanted to play with Lily. He said, "Can I see your teddy bear? He is so soft and cute. Can I see it?"

Lily shook her head and said, "No, Tom. This is my teddy bear. You have your own toys and I own them. Go away."

Tom felt sad and angry. He wanted to play with the teddy bear too. He said, "Lily, you are mean. You do not like me. I do not want to play with you. Give me your teddy bear. I want to play with it."

Lily did not listen. She ran to her mom and told her what Tom said. Mom was not happy. She said, "Tom, that is not nice. You should not take Lily's teddy bear. You should say sorry and make up. You should find a way to share. Do you understand?"

Tom was

---

## Prompt: "One day the sky turned green"

### Baseline (30M) (step 8000)

One day the sky turned green and sparkled in the dark sky. The animals all flew around looking for something to do. Then a little monkey came up to a tree. He saw a rope hanging on the branches. He thought it would be an easy way to use the rope.

The monkey said to the rope, "Why do you want to use it?" 

The rope replied, "I want to use it for a fun day. I'm going to have an adventure!" 

The monkey thought for a moment and then said, "Okay, but be careful. You don't want to make a mess." 

The monkey was very happy and he played with the rope. He pretended to be a monkey. The animals all cheered and played together while they were having fun. The monkey was so happy he had brought the rope with him.

### Social Field (31M) (step 8000)

One day the sky turned green and the little bird had an idea. She wanted to fly high up in the sky and see the beautiful flowers. She asked her friends, the birds, to help her, and they agreed.

The little bird flew to the sky and started to fly. She flew so high that the clouds were falling down. The little bird felt happy and content. The wind kept the sky for a long time.

When the sun started to go down, the little bird's friends came to her house. They told her how happy she was and taught her a lesson about climbing trees. The little bird was so happy to have her friends to play with. They all promised to come back at the big tree every day.

### Baseline 1.8x (53M) (step 10000)

One day the sky turned green and orange, which made the boy very happy. He wanted to make the sky look more beautiful, so he started to make a plan. He wanted to make the clouds look beautiful and bright. 

So the boy gathered all the things he needed and made a big, bright rainbow. He ran around, laughed and smiled. 

He created the most beautiful rainbow ever! Everyone was amazed and clapped for him. 

The boy was so proud of his work that he ran inside to show his mom. She was very pleased and gave him a big hug. She said, "You did a great job! I'm so proud of you!" 

The boy smiled and hugged her back. He was so happy that he made the sky look so beautiful.

---

## Prompt: "The old wizard opened his book and"

### Baseline (30M) (step 8000)

The old wizard opened his book and said, "Hello! Do you like my books?" The little girl smiled and said, "Yes, I do! They are the best!"

The old wizard said, "I am grateful for your help. Would you like to try some of my books?" The little girl nodded and they spent the rest of the day reading books together.

### Social Field (31M) (step 8000)

The old wizard opened his book and started to read.

"What are you reading?" asked the wizard.

"I'm reading a book," said the wizard.

The wizard smiled and said, "That's great! I like reading too much."

The wizard spent the rest of the day reading more books and enjoying the stories. He thanked the wizard for the lovely book and said goodbye to his new friend.

### Baseline 1.8x (53M) (step 10000)

The old wizard opened his book and saw that the pages were full of pictures. He was very happy and began to read the words.

Afterwards, the old wizard smiled and thanked the old wizard for his words.

The old wizard had been very kind and he smiled back and said, "That was nice of you to help me,"

The little boy nodded and said, "Yes, I am very grateful."

The old wizard smiled back and said, "We have come to visit you anytime, and I'm always grateful for your kindness".

The little boy smiled back and said, "I would love that too!"

The old wizard waved goodbye and went off into the night.

---

*Training ~58-75% complete.*
