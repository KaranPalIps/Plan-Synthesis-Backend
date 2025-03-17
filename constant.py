question = """
Please provide the following information in JSON format. Each key should represent a question, and each value should be the corresponding answer:

1. What is the last review date?
2. What is the effective date?
3. What are the HCPCS codes?
4. What is the policy number?
5. What is the policy name?
6. What is the email address?
7. What is the fax number?
8. What is the address?
9. What is the phone number?
10. What is the length of approval?

Please ensure the format is strictly JSON, with proper structure and punctuation.
JSON format example:
{
    "What is the last review date?": "2021-08-01",
    "What is the effective date?": "2021-08-01",
    "What are the HCPCS codes?": ["A1234", "B5678"],
    "What is the policy number?": "123456",
    "What is the policy name?": "Policy ABC",
    "What is the email address?": "xyz@#mail.com",
    "What is the fax number?": "123-456-7890",
    "What is the address?": "123 Main St, City, State, Zip",
    "What is the phone number?": "123-456-7890",
    "What is the length of approval?": "1 year"
}
Stricly adhere to the JSON format and structure to ensure accurate processing.
"""