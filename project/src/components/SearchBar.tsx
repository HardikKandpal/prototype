import React, { useState } from "react";
import { propertyApi } from "../api/propertyApi";

export function SearchBar({ onSearchResults }: { onSearchResults: Function }) {
  const [searchParams, setSearchParams] = useState({
    location: "",
    min_price: "",
    max_price: "",
    bedrooms: "",
    bathrooms: "",
    total_area: "",
    has_balcony: false,
    sort_by: "price",
    sort_order: "asc",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setSearchParams({ ...searchParams, [e.target.name]: e.target.value });
  };

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchParams({ ...searchParams, has_balcony: e.target.checked });
  };

  const handleSearch = async () => {
    const params = { ...searchParams };

    // Convert empty values to undefined to avoid sending them as filters
    Object.keys(params).forEach((key) => {
      if (params[key] === "" || params[key] === null) {
        delete params[key];
      }
    });

    const results = await propertyApi.search(params);
    onSearchResults(results);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h3 className="text-2xl font-semibold">Find Your Property</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
        <input
          type="text"
          name="location"
          placeholder="Location"
          value={searchParams.location}
          onChange={handleChange}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          name="min_price"
          placeholder="Min Price"
          value={searchParams.min_price}
          onChange={handleChange}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          name="max_price"
          placeholder="Max Price"
          value={searchParams.max_price}
          onChange={handleChange}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          name="bedrooms"
          placeholder="Bedrooms"
          value={searchParams.bedrooms}
          onChange={handleChange}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          name="bathrooms"
          placeholder="Bathrooms"
          value={searchParams.bathrooms}
          onChange={handleChange}
          className="px-4 py-2 border rounded-lg"
        />
        <input
          type="number"
          name="total_area"
          placeholder="Min Area (sq ft)"
          value={searchParams.total_area}
          onChange={handleChange}
          className="px-4 py-2 border rounded-lg"
        />
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={searchParams.has_balcony}
            onChange={handleCheckboxChange}
            className="form-checkbox h-5 w-5"
          />
          <span>Has Balcony</span>
        </label>
        <select name="sort_by" value={searchParams.sort_by} onChange={handleChange} className="px-4 py-2 border rounded-lg">
          <option value="price">Sort by Price</option>
          <option value="total_area">Sort by Area</option>
          <option value="bedrooms">Sort by Bedrooms</option>
        </select>
        <select name="sort_order" value={searchParams.sort_order} onChange={handleChange} className="px-4 py-2 border rounded-lg">
          <option value="asc">Low to High</option>
          <option value="desc">High to Low</option>
        </select>
      </div>
      <button className="mt-6 bg-blue-600 text-white px-6 py-3 rounded-lg w-full" onClick={handleSearch}>
        Search Properties
      </button>
    </div>
  );
}
