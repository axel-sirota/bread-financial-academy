#!/usr/bin/env python3
"""
Generate call transcripts for Week 6 Call Center ML lab.

Creates call_transcripts.json with pre-computed transcripts as backup
for Amazon Transcribe (in case transcription jobs don't complete in time).

Usage:
    python generate_transcripts.py

Output:
    call_transcripts.json
"""

import json

# Call scripts - dialogues between agent and customer
CALL_SCRIPTS = {
    "call_001": {
        "title": "Fraud Dispute - Unauthorized Charge",
        "duration_seconds": 145,
        "sentiment": "NEGATIVE",
        "is_fraud": True,
        "dialogue": [
            ("agent", "Thank you for calling First National Bank, this is Sarah speaking. How can I help you today?"),
            ("customer", "Yes, I'm calling about a charge on my account that I didn't make. I need to dispute it immediately."),
            ("agent", "I understand. I'm sorry to hear that. Can you please provide your account number so I can look into this?"),
            ("customer", "It's 4532-7891-2345-6789. The charge is for eight hundred forty seven dollars and ninety nine cents."),
            ("agent", "Thank you. I see the charge from Global Electronics for $847.99 on November 12th. Can you tell me more about why you're disputing this?"),
            ("customer", "I never made that purchase. I've never even heard of Global Electronics. I think someone stole my card information. This is really frustrating."),
            ("agent", "I completely understand your frustration, and I'm very sorry this happened to you. I'm going to flag this as potential fraud and start an investigation right away."),
            ("customer", "Thank you. What happens next? Will I get my money back? I'm really worried about this."),
            ("agent", "Yes, absolutely. I've placed a temporary hold on your card for security and initiated the dispute process. You should receive a provisional credit within 24 hours, and a new card will arrive within 3 to 5 business days."),
            ("customer", "Okay, thank you for your help Sarah. I appreciate you taking this seriously."),
            ("agent", "Of course. Is there anything else I can help you with today?"),
            ("customer", "No, that's all. Thank you again."),
            ("agent", "You're welcome. Thank you for calling First National Bank. Have a good day."),
        ]
    },
    "call_002": {
        "title": "Account Balance Inquiry",
        "duration_seconds": 75,
        "sentiment": "NEUTRAL",
        "is_fraud": False,
        "dialogue": [
            ("agent", "Good afternoon, First National Bank, this is Michael. How may I assist you?"),
            ("customer", "Hi, I just want to check my account balance please."),
            ("agent", "Certainly. May I have your account number?"),
            ("customer", "Sure, it's 9876-5432-1098-7654."),
            ("agent", "Thank you. I see your checking account has a current balance of two thousand three hundred forty five dollars and sixty seven cents."),
            ("customer", "Okay, that sounds right. And what about my savings?"),
            ("agent", "Your savings account shows a balance of eight thousand nine hundred twelve dollars."),
            ("customer", "Perfect, thank you. That's all I needed."),
            ("agent", "You're welcome. Is there anything else I can help you with today?"),
            ("customer", "No, that's it. Thanks."),
            ("agent", "Thank you for calling First National Bank. Have a great day."),
        ]
    },
    "call_003": {
        "title": "Lost Card Report",
        "duration_seconds": 120,
        "sentiment": "NEGATIVE",
        "is_fraud": True,
        "dialogue": [
            ("agent", "First National Bank, this is Jennifer. How can I help you?"),
            ("customer", "I lost my debit card! I think it might have been stolen. I need to cancel it right now!"),
            ("agent", "I understand this is stressful. Let me help you right away. Can you provide your account number?"),
            ("customer", "Yes, it's 1234-5678-9012-3456. Please hurry, I'm really worried someone is using it."),
            ("agent", "I'm blocking the card right now as we speak. The card ending in 3456 is now deactivated. No one can use it."),
            ("customer", "Oh thank goodness. Have there been any charges I didn't make?"),
            ("agent", "Let me check your recent activity. I see two charges in the last hour - one at a gas station for forty five dollars and one at an electronics store for three hundred twenty dollars. Do you recognize these?"),
            ("customer", "No! I haven't used my card today at all. I've been at home all day. Those aren't mine!"),
            ("agent", "I understand. I'm marking both of these as fraudulent charges right now and starting a dispute. You'll receive a full refund within 24 hours."),
            ("customer", "Thank you so much. When will I get a new card?"),
            ("agent", "I'm ordering a replacement card with expedited shipping. You'll receive it within 2 business days. Is your mailing address still 456 Oak Street, Phoenix, Arizona?"),
            ("customer", "Yes, that's correct. Thank you for helping me so quickly."),
            ("agent", "Of course. Stay safe, and call us if you have any other concerns."),
        ]
    },
    "call_004": {
        "title": "Positive Follow-up - Issue Resolved",
        "duration_seconds": 90,
        "sentiment": "POSITIVE",
        "is_fraud": False,
        "dialogue": [
            ("agent", "Thank you for calling First National Bank, this is David. How can I help you today?"),
            ("customer", "Hi David! I'm calling to say thank you, actually. I had an issue with a duplicate charge last week and your colleague Maria helped me sort it out."),
            ("agent", "That's wonderful to hear! I'm glad we could resolve that for you. I can see in your account notes that Maria processed a refund for the duplicate transaction."),
            ("customer", "Yes, exactly. The refund came through yesterday and everything looks correct now. I just wanted to call and let you know how happy I am with the service."),
            ("agent", "Thank you so much for taking the time to call and share that feedback. I'll make sure Maria gets recognized for her excellent work."),
            ("customer", "Please do! She was so patient and helpful. I've been a customer for ten years and this is why I stay with First National."),
            ("agent", "We really appreciate your loyalty. Is there anything else I can assist you with today?"),
            ("customer", "No, that's all. Just wanted to share the positive feedback. Have a great day!"),
            ("agent", "You too! Thank you for being a valued customer."),
        ]
    },
    "call_005": {
        "title": "International Transfer Inquiry (Spanish Speaker)",
        "duration_seconds": 110,
        "sentiment": "NEUTRAL",
        "is_fraud": False,
        "dialogue": [
            ("agent", "First National Bank, this is Ana speaking. How may I help you?"),
            ("customer", "Hola, buenos días. I need to make a wire transfer to Mexico. Can you help me with that?"),
            ("agent", "Of course! I'd be happy to help you with an international wire transfer. May I have your account number please?"),
            ("customer", "Yes, my account is 5555-4444-3333-2222."),
            ("agent", "Thank you. I see your account. To send an international wire to Mexico, I'll need the recipient's full name, their bank name, and the CLABE number which is the Mexican bank account number."),
            ("customer", "Okay, the name is Carlos Rodriguez Martinez, the bank is Banco Nacional de México, and the CLABE is 012 180 001234567890."),
            ("agent", "Perfect. And how much would you like to send?"),
            ("customer", "I want to send one thousand five hundred dollars. How much will the fee be?"),
            ("agent", "The wire transfer fee is thirty five dollars, and the exchange rate today is approximately 17.2 pesos per dollar. Your recipient would receive about twenty five thousand eight hundred pesos."),
            ("customer", "That sounds good. Please proceed with the transfer."),
            ("agent", "I've initiated the transfer. It will arrive within 1 to 2 business days. You'll receive a confirmation email shortly. Is there anything else?"),
            ("customer", "No, muchas gracias. Thank you for your help."),
            ("agent", "De nada. Thank you for choosing First National Bank. Have a great day!"),
        ]
    }
}


def format_transcript(dialogue: list) -> str:
    """Format dialogue into a single transcript string."""
    lines = []
    for speaker, text in dialogue:
        speaker_label = "Agent" if speaker == "agent" else "Customer"
        lines.append(f"{speaker_label}: {text}")
    return "\n\n".join(lines)


def main():
    """Generate call transcripts JSON file."""
    output = {}

    print("Generating call transcripts...")

    for call_id, data in CALL_SCRIPTS.items():
        transcript = format_transcript(data["dialogue"])

        output[call_id] = {
            "title": data["title"],
            "transcript": transcript,
            "duration_seconds": data["duration_seconds"],
            "expected_sentiment": data["sentiment"],
            "is_fraud_case": data["is_fraud"],
            "speakers": [
                {"speaker": "spk_0", "role": "agent"},
                {"speaker": "spk_1", "role": "customer"}
            ],
            "word_count": len(transcript.split())
        }

        print(f"  {call_id}: {data['title']} ({data['duration_seconds']}s, {data['sentiment']})")

    # Write to JSON
    output_file = "call_transcripts.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nGenerated {len(output)} call transcripts")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
