const BASE_URL = "http://localhost:5000/api";

export const propertyApi = {
  search: async (searchParams: object) => {
    const response = await fetch(`${BASE_URL}/property-search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(searchParams),
    });
    return response.json();
  },

  getPropertyDetails: async (id: number) => {
    const response = await fetch(`${BASE_URL}/property/${id}`);
    return response.json();
  },

  getRecommendations: async (id: number) => {
    const response = await fetch(`${BASE_URL}/recommendations/${id}`);
    return response.json();
  },

  estimateValue: async (propertyData: object) => {
    const response = await fetch(`${BASE_URL}/estimate-value`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(propertyData),
    });
    return response.json();
  },

  generateDescription: async (details: string) => {
    const response = await fetch(`${BASE_URL}/generate-description`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ details }),
    });
    return response.json();
  },

  analyzeImage: async (imageFile: File) => {
    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await fetch(`${BASE_URL}/analyze-image`, {
      method: "POST",
      body: formData,
    });

    return response.json();
  },
};

