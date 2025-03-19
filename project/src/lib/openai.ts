import OpenAI from 'openai';

const apiKey = import.meta.env.VITE_OPENAI_API_KEY;

if (!apiKey) {
  throw new Error(
    'OpenAI API key is missing. Please add VITE_OPENAI_API_KEY to your .env file'
  );
}

// Initialize OpenAI client
export const openai = new OpenAI({
  apiKey: apiKey,
  dangerouslyAllowBrowser: true // Note: In production, API calls should go through your backend
});

export async function generatePropertyDescription(details: {
  type: string;
  bedrooms: number;
  bathrooms: number;
  size: number;
  location: string;
  features: string[];
}): Promise<string> {
  const prompt = `Generate a compelling real estate description for a ${details.type} with ${details.bedrooms} bedrooms, 
    ${details.bathrooms} bathrooms, ${details.size} square feet, located in ${details.location}. 
    Key features include: ${details.features.join(', ')}.`;

  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "You are a professional real estate copywriter who creates engaging and accurate property descriptions."
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0.7,
    max_tokens: 350
  });

  return response.choices[0].message.content || '';
}

export async function analyzePropertyImage(imageUrl: string): Promise<{
  features: string[];
  condition: string;
  style: string;
}> {
  const response = await openai.chat.completions.create({
    model: "gpt-4-vision-preview",
    messages: [
      {
        role: "system",
        content: "You are a real estate image analysis expert. Analyze the property image and provide key features, condition, and architectural style."
      },
      {
        role: "user",
        content: [
          { type: "text", text: "Analyze this property image and provide key features, condition, and architectural style." },
          { type: "image_url", image_url: imageUrl }
        ]
      }
    ],
    max_tokens: 150
  });

  // Parse the response into structured data
  const analysis = response.choices[0].message.content || '';
  // This is a simplified parsing, you might want to make it more robust
  return {
    features: analysis.split(',').map(f => f.trim()),
    condition: 'Excellent', // You would extract this from the analysis
    style: 'Modern' // You would extract this from the analysis
  };
}

export async function estimatePropertyValue(details: {
  location: string;
  type: string;
  size: number;
  bedrooms: number;
  bathrooms: number;
  yearBuilt: number;
  features: string[];
}): Promise<number> {
  const prompt = `Estimate the market value for a ${details.type} property with the following details:
    - Location: ${details.location}
    - Size: ${details.size} sq ft
    - Bedrooms: ${details.bedrooms}
    - Bathrooms: ${details.bathrooms}
    - Year Built: ${details.yearBuilt}
    - Features: ${details.features.join(', ')}
    
    Provide the estimated value in USD.`;

  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "You are a real estate valuation expert. Provide property valuations based on given details."
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0.2,
    max_tokens: 100
  });

  // Extract the numerical value from the response
  const valueText = response.choices[0].message.content || '';
  const value = parseFloat(valueText.replace(/[^0-9.]/g, ''));
  return value || 0;
}

export async function processNaturalLanguageSearch(query: string): Promise<{
  type?: string;
  priceRange?: { min: number; max: number };
  location?: string;
  features?: string[];
  bedrooms?: number;
  bathrooms?: number;
}> {
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "You are a real estate search expert. Extract structured search parameters from natural language queries."
      },
      {
        role: "user",
        content: query
      }
    ],
    temperature: 0.3,
    max_tokens: 150
  });

  // Parse the response into structured search parameters
  // This is a simplified version, you would want to make it more robust
  const searchParams = JSON.parse(response.choices[0].message.content || '{}');
  return searchParams;
}
